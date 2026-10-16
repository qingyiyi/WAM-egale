# 投机验证器系统优化、Fast-WAM迁移与“双需求”解释

2026-10-11T23:26:03+08:00（Asia/Shanghai）；投机采样讨论记录，**非tile算子里程碑**。本次完成只读源码/文献/既有元数据检查后，按用户长期约定追加HW知识；没有新增GPU、训练、kernel实现或公共源码变更。

## 1. 用户提出的研究假设

用户认为投机把大部分计算集中到Transformer verifier，可用mega Transformer加速；询问Fast-WAM是否也能加入去噪方向投机，以及长历史双需求慢化原因和“双需求”含义。这里将mega Transformer理解为**同一冻结模型的融合/持久化执行实现**，不是扩大target模型规模。

判断：集中target方向求值到批量验证提供了算法—系统协同机会，值得保留此思路。但当前还没有证明纯算力是唯一瓶颈、没有实现mega kernel，也不能以更多冗余候选计算作为收益本身。优化目标是有效提交动作的真实净成本及闭环任务行为。

## 2. 双需求的确切含义

依据 [固定合同](../../../review/OMISSIONS.md) 第73–92、121–155行及 [监督器](../../../review/OMISSIONS.md) 第75–86、163–169、203–208行：

- 两个独立模型worker各加载一份冻结target和A_off，均位于同一张A100 CUDA_VISIBLE_DEVICES=0，各有独立进程、缓存和请求。
- **单需求**：两份模型都保持驻留，只让primary worker处理一个请求，peer空闲。
- **双需求**：primary和peer同时恢复同一fixture、运行同一算法；T/T或A_off/A_off，不是T与A_off混跑。原policy/action同步门让动作计算窗口重叠，正式比较要求重叠比例至少90%。
- init3/4是两份fixture来源，不是每个双需求槽同时混用两个不同init；peer复制当前primary的fixture，不充当独立episode。
- 双需求不是K2B2的分支宽度B=2，不是CFG conditional/unconditional两行，不是两张GPU，不是大小模型协同/联训；完整episode数量仍是0。

这是一项**同卡并发压力检查**，不代表实际单机器人一定同时跑两个模型。单需求时也有两份模型驻留，所以单需求结果不能直接当单份模型独占所有显存的部署结果。

## 3. 长历史慢化：已知事实与未隔离的原因

[正式GPU成本结果](FIXED-LOAD-COST-RESULT-20261011-213036.md)显示：

| 长历史条件 | init3 T / A_off(s) | init4 T / A_off(s) | A_off相对T |
|---|---:|---:|---:|
| 单需求 | 4.7442 / 4.9651 | 4.5754 / 5.0867 | 慢4.7% / 11.2% |
| 双需求 | 8.1618 / 10.6287 | 8.2988 / 10.6400 | 慢30.2% / 28.2% |

同一长fixture下，双/单耗时比T为1.72/1.81，A_off为2.14/2.09。双需求本来就增加同卡工作需求；应看A_off相对T的额外敏感性，而不是要求两请求同时执行时单请求耗时不变。这些是每个请求的动作分支延迟，不是已测得的最优服务吞吐或自然episode E2E。

同fixture跨需求模式的tree、scheduler更新、target行量计数一致，因此不能把**长双相对长单**变慢归因为接受率下降或多做了去噪更新。与短历史比较则仍混有不同状态/接受路径，不能声称KV长度是唯一原因。

长历史单需求已经没有净收益，所以问题不是仅“解决双需求干扰”。建议以实际部署单需求为主、双需求为压力辅检，区分 verifier固有工作扩张与并发资源/host调度效应。

## 4. CPU源码检查发现的具体优化线索

### 4.1 候选KV被每层实物化复制

[WanAttention](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py) 第633–660行先从原缓存取有效history，再用 `index_select` 为每个候选复制history K/V，随后 `cat` 当前候选K/V后进入attention。CFG行映射在 [server](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py) 第747–779行为7个conditional候选映射row0、7个unconditional映射row1；**只有两份不同history，却物化14份候选history**。`index_select`不与原输入共享存储，见 [PyTorch2.9官方语义](https://docs.pytorch.org/docs/2.9/generated/torch.index_select.html)。

事后CPU检查已有8个primary ON条件的全部candidate layer元数据，Q为[14,16,24,128]，K/V各为[14,288或2160,24,128]，BF16。扣除16当前action token后，history长度272或2144。

| 单层单次verifier的K+V输出payload | 短历史 | 长历史 |
|---|---:|---:|
| 两份不同CFG history | 6.375MiB | 50.25MiB |
| index_select后14份history | 44.625MiB | 351.75MiB |
| cat后完整candidate K/V | 47.25MiB | 354.375MiB |

这些是形状推导的**输出张量payload**，不是实测HBM流量、峰值显存、分配次数或性能因果占比；两输出存在嵌套生命周期，不能简单加成“峰值显存”。当前每个verifier有30层，且每chunk约26–29个verifier batch，这一重复过程值得优先检查。

候选系统方案是共享prefix KV、为candidate独立保留当前K/V、避免物化整段history，并保证候选之间仍完全隔离；不能直接把候选tokens拼进同一序列使其互相attention。共享存储也不意味着attention运算或HBM读取天然只做一次。方案尚未实现，也不混入tile路线既有收益。

### 4.2 存在host-device同步线索

同一attention里有CUDA `mask.nonzero`、candidate row范围校验的 `bool(tensor.all())`；server候选sigma/timestep准备和原scheduler还有CPU取标量。CUDA nonzero会同步host/device，见 [PyTorch2.9官方说明](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html)。这些是现有代码事实/潜在串行点，**尚未测定其OFF耗时占比**。

普通T路径第677行同样使用nonzero，不能把所有同步成本只归给A_off，更不能据此宣布根因已确认。仍缺同步/kernel launch、GEMM、attention、gather/index_select/cat、HBM/SM占用及双进程排队的受控分解。

## 5. “只有Transformer、没有scheduler”的适用边界

狭义 [_target_action_candidate_directions](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py) 第1906–1942行的昂贵求值确实是一次batched Transformer；但包含输入准备、Transformer内部KV/attention处理及CFG后处理，不等于只含GEMM。

完整K2B2循环第1042–1061、1195–1237行仍用原scheduler构建树/回退/提交，并有draft、pack、评分和选择。已提交更新仍50次，长fixture实际tree scheduler调用192/193次；并未删除去噪求解。ON诊断verifier forward占动作69%–84%，draft约2%–5%，说明重心在verifier，但这些并非clean OFF kernel占比。长双init3 ON scheduler约0.082s/action11.089s仅为该次诊断线索，不做OFF−ON或嵌套phase重加。

**我们减少的是昂贵target前向的串行依赖，利用draft预构建候选；不是宣称所有计算量降低或scheduler消失。** 51→29–32次target前向，同时102→370–412 CFG行调用，这个代价必须由执行效率/有效接受抵消。

## 6. Fast-WAM可以叠加方向投机，scheduler不是结构障碍

[Fast-WAM论文](https://arxiv.org/html/2603.16666v2)采用single-pass world encoder和10步action flow。Fast-WAM本身没有本项目意义的verify过程，它是可加入投机的target模型；不能把它的普通去噪循环称作既有投机验证器。

进一步核对 [官方FastWAM固定commit源码](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py) 的infer_action：先prefill_video_cache，循环调用action expert，再调用scheduler；已支持compiled prefill/denoise core。实际步数由调用/评测配置传入：论文为10，低层方法默认参数是20，不能把默认签名当评测设置。

[scheduler同commit源码](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py) 第63–88行预构建timesteps/deltas，step核心是sample + model_output × delta；scheduler不是同Transformer一样的大计算，也不是天然无法融合/向量化的黑盒。

**推断的迁移方案，尚未实现/验证：**保留Fast-WAM原视觉编码和缓存，draft未来action velocity/方向，用原sigma/delta构造候选state，再批量调用真实action expert验证并回退。迁移后狭义target求值同样可只执行Transformer，将scheduler放在构建/提交侧。需要重新对齐action形状、proprio/text条件、时间表、mask/cache、候选独立性、阈值尺度与draft监督；不能直接复用LingBot的112维checkpoint或原阈值，也不能假定K2B2/7候选一定是好预算。

10步相对50步摊销空间更小、少量anchor/tail占比更大，但“计算较少”本身不是劣势。应验证更窄candidate是否已有净收益，并和Fast-WAM**已compiled/缓存优化**原模型公平比较；原T也有权使用适用的kernel优化。参考 [官方当前加速入口说明](https://github.com/yuantianyuan01/FastWAM)，不把它不同硬件的毫秒数字与本项目直接相除。

## 7. mega Transformer思路与接下来的关卡

值得探索的是：批量verification提供稳定执行区域，是否可用共享KV、融合Norm/RoPE/残差、小算子/执行图等减少搬运和host/kernel调度，再评估更大范围block或persistent执行。**不是先承诺30层全部放进一个kernel一定更快**；若主瓶颈是history重复搬运，仅减少launch可能不够，且融合不能自动消除冗余候选本身的工作。

算法（draft/候选预算/质量）与系统（同模型执行效率）要分别消融；任何对T同样适用的优化也要给T，闭环质量另关验证，不能用kernel收益冒充投机独有收益。mega kernel在本文件只作为投机验证器研究假设；不自动调用或归入tile Milestone。

下一步建议仅CPU：整理共享KV/同步/候选独立性的数据流和最小比较合同，优先当前单需求长历史无收益，再以双需求验证资源敏感性；同时完成Fast-WAM接口可迁移性清单。随后才单独确认最小GPU拆解/实现预算，不追加完整episode，不直接训练或改阈值/bonus。当前根因仍未隔离，success/环境步数/E2E目标也未解决。

[CPU证据](../../../review/OMISSIONS.md)；[研究主入口](RESEARCH.md)。
