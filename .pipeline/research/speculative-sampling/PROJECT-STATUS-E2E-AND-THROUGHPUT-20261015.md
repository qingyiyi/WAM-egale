# 投机 WAM 项目完整汇报：E2E、进展、实验吞吐与后续计划

> **17:41新结果与后续入口：** [本机2→3→4报告](LOCAL-234-RESULT-20261015.md)：reset八完整任务严格配对已通过；数据179928行全量审核和TF/MLP六组公平训练已完成；四臂40完整DEV运行中；来源更正、CPU后处理失败恢复、空间及AutoDL分流均记录。本文15:56的资源状态、数据“待审核”和reset“未在线验收”等保留为历史快照；旧正式速度仍有效，新TF/bank暂无正式计时。

2026-10-15，Asia/Shanghai。本文承接本轮会话，已完成项均引用原始结果；新增工作包括CPU账本复算、原T5文本bank构建及缓存/裁padding真实CPU对照、运行状态读取、文献核查和HW整理。运行状态截至15:56；不同吞吐/阶段表保留各自明确标注的采样时点，终态以job文件为准。不是论文初稿。

> 15:28追加：已直接拆出AutoDL前17条的客户端请求时间与生命周期。平均每条384.3秒，其中模型reset233.0秒（60.6%）、policy95.8秒（24.9%）、cache7.1秒（1.9%）、环境/本地reset/客户端余项15.2秒（4.0%）、任务外启动准备/drain33.2秒（8.6%）。不能将本机reset46.8%直接套在远端；详见第6节。15:27时进度17/100。

## 1. 当前目标与结论

目标是在冻结 LingBot-VA/LIBERO-Long target 的情况下，用小 draft 提议连续 action-flow 更新，批量 target verifier 接受合法路径并回退，使任务成功率保持、环境步数和长尾不过度增加，同时降低完整 policy chunk 和完整任务墙钟。另行研究 verifier 的 MegaTransformer 系统优化；原 tile Transformer 路线与 HW 保留。

目前**实现了推理加速，还没有完成最终 Transformer draft 选型和 Long10 质量证明**。冻结的运行参照仍是 A_off MLP、K2/B2 bonus、`prefix_then_cost`、阈值 `0.007839680526450805`、共享历史 KV、O1/O2/O3。这是可复现的当前候选，不等于投机研究已经收尾。

最清楚的速度证据是：短/中/长 KV 的完整 policy chunk 分别 **1.602× / 1.544× / 1.500×**；新四任务配对的任务运行 **1.151×**；计入冷 server 启动至 client 退出 **1.091×**。前一项不含 reset，后两项不能混用。[chunk 结果](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)、[正式四任务结果](../../../review/OMISSIONS.md)

## 2. “E2E”到底包括哪些时间

| 口径 | 起止与内容 | 当前结果 | 边界 |
| --- | --- | --- | --- |
| 完整 policy chunk | 同一真实输入开始 infer，直到 video/action、draft、verify、接受/回退、输出完成 | 短/中/长 1.602×/1.544×/1.500× | 不含每回合 reset、仿真执行或模型加载 |
| 任务运行 E2E，即原 client `elapsed_sec` | 本地环境 reset/稳定、模型 reset 请求、全部 policy/cache 请求、执行至成功或原 horizon | `1033.403/898.071=1.150692×` | 不含 EGL 环境构造/关闭和模型加载 |
| 冷启动 E2E | 冷 server 启动到 client 退出，包含模型准备、环境构造及任务执行 | `1192.672/1093.244=1.090947×` | 任务后的异步存档 drain/清理另列，不是整批调度器总时长 |

因此上次只称“完整 E2E”的表述需要明确为**模型已加载后的任务运行 E2E**。论文中应同时列出上述口径，不能让读者把 1.151× 理解为从程序启动到所有进程回收。

四任务正式计划为 task0/1/5/9 各 init29，T/A_off 顺序交错，8 条完整轨迹全部完成；初态、XML、初始观测及 RNG 配对审核通过，496 条 GPU 遥测无外来进程，目标调用计数闭合。T/A_off 均 4/4 成功，环境步数合计 1054/1044，policy chunk 为 69/68。每个任务只有一个初态，速度正向不构成总体成功率非劣证明。

| task | T / A_off 环境步数 | T / A_off 自然任务秒 | 速度比 |
| --- | ---: | ---: | ---: |
| 0 | 287 / 272 | 275.223 / 220.846 | 1.2462× |
| 1 | 264 / 262 | 258.022 / 218.339 | 1.1818× |
| 5 | 194 / 185 | 224.779 / 210.009 | 1.0703× |
| 9 | 309 / 325 | 275.380 / 248.877 | 1.1065× |

另外，task0/init25–28 的单任务四对已正式审计：T 3/4、A_off 4/4，双方都成功的三对合计 1.188861×。第4对 T 失败、轨迹很长，全部四对比值 1.4669× 不能解释成投机性能收益。它与新四任务结果分别报告，不挑选失败来放大加速。[单任务原始分析](../../../投机采样/results/runtime/p0-formal-e2e-speed-v1-single-r2-20261015/FINAL-ANALYSIS.json)

## 3. 为什么任务 E2E 只有约 1.15×

下面是同一批四任务的互不重叠耗时账，单位为四条轨迹合计秒。请求时间包含传输与服务端执行，不等于 GPU busy time。

| 部分 | 原 T | A_off | T 中占比 | A_off−T |
| --- | ---: | ---: | ---: | ---: |
| 模型 reset 请求 | 483.740 | 505.138 | **46.81%** | +21.397 |
| 完整 policy 请求 | 456.802 | 298.572 | **44.20%** | **−158.230** |
| 观测/cache 请求 | 29.111 | 27.935 | 2.82% | −1.176 |
| 仿真执行动作 | 42.949 | 44.190 | 4.16% | +1.240 |
| 本地 reset/稳定 | 20.704 | 22.133 | 2.00% | +1.429 |
| 客户端余项 | 0.097 | 0.104 | <0.01% | +0.007 |
| 合计 | **1033.403** | **898.071** | 100% | **−135.332** |

policy 请求自身已有 **1.530×**，但它只覆盖原总耗时的 44.2%。节省的 158.2 秒又被 reset 等部分增加的约 22.9 秒抵消，净省 135.3 秒，即总时间少 13.10%、速度 1.151×。reset 代码没有因投机发生算法变化，不能把本轮 +21.4 秒归为投机必然增加的计算；它体现了当前批次 reset 墙钟差异，子阶段尚未 profile。

源码可确认 reset 会清 transformer/VAE 缓存、创建新 KV，并重新执行 `encode_prompt`；`enable_offload=True` 时 T5 位于 CPU。当前 CFG 还需要空负提示的编码。**这是优先检查的重复工作来源，但尚无 reset 内部计时，不能把全部约 2 分钟都归给 T5**。[reset 源码](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)、[T5 加载](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)、[计时范围源码](../../../投机采样/experiments/formal_e2e_speed_v1/client_entry.py)

以本批 T 的占比作固定轨迹、固定其他耗时的 Amdahl 算术：`S_episode=1/(0.558+0.442/S_policy)`。policy 2×/3×/4× 时，任务分别约 1.284×/1.418×/1.496×；policy 耗时趋零的上限也仅 1.792×。这些是条件计算，不是新实测。若把两臂的整个模型 reset 都理想化为零，本批剩余时间比也只有约 1.399×，所以论文必须报告完整成本，不能直接把单 chunk 1.5×写成任务 1.5×。

冷启动进一步摊薄收益：每条 T/A_off 在自然任务外平均多 39.8/48.8 秒，合计后得到 1.091×。这部分包含模型/服务和 client/EGL 准备，不能全部叫模型权重读取。每条轨迹都重新启动进程的实验组织也降低了采集吞吐。

还有 verifier 本身的成本：一次批量验证有7逻辑候选、14个CFG行，并非一次普通 target 状态的免费替代。四任务中 1330 棵树的双层接受为907棵（68.2%），其余需回退；共享KV节省了历史attention复制和计算，cross-attention/投影等仍有候选工作量。真实 profile 的 cross-attention 占唯一 CUDA kernel 累计时长54.40%–62.22%，不是完整 E2E 的同等占比，不能直接拿它推算融合收益。[profile 证据](DRAFT-DATA-VERIFIER-RESULT-20261014.md)

CPU 复算及图：[机器可读账本](../../../review/OMISSIONS.md)、[复算源码](../../../review/OMISSIONS.md)、[耗时图](../../../review/OMISSIONS.md)。

## 4. 本轮会话承接的完整优化与实验账

下表各行基线和工作负载不同，收益**不能相乘**；旧数值验收也不自动证明新的近似算法任务质量。

| 工作 | 做了什么、实测结果 | 当前采用/未完成 |
| --- | --- | --- |
| 基线与合同对齐 | 真实 Long checkpoint、20 video/50 action 更新、观测/动作、初态、KV 生命周期、RNG、成功/800步 horizon 与计时端点对齐 | 当前实验基础；旧 HW M01–M04 完成不代表投机研究完成 |
| K2/B2 树与接受路径 | 轻量 MLP 提议两层、每层两候选，七节点批量 target 验证；最长合法前缀后比较成本，失败走 target | 当前采用；阈值来自旧误差 q90，不是90%成功概率 |
| Bonus | 双层均接受时消费本次 verifier 已算出的叶 target 方向，多提交一步；尾段保持完整50更新 | 当前采用。旧两个长KV状态 OFF→BONUS 1.157–1.168×；该轮相对OFF任务仅1.018×，不能只报推理收益 |
| 共享历史 self-attention KV | 七候选读取相同历史，避免按候选复制；固定完整verifier 5.171→3.360秒，下降35.02%；额外峰值891→257MiB | 当前采用。不是整模型总显存或全任务35%收益 |
| O1/O2/O3 | 复用Python模板、减少标量同步、复用工作区；旧两个完整任务707.183→642.318秒，下降9.17%，输出/动作轨迹一致 | 当前采用；相对旧S0，不能与当前对T结果相乘 |
| 当前完整 chunk 正式测试 | 短/中/长KV 1.602×/1.544×/1.500×，18/18配对更快；同算法输出/route/KV/RNG检查通过 | 已完成；不含reset，三状态重复不是18个不同任务 |
| 当前任务正式测试 | 单task双方成功3对1.189×；新四task双方4/4成功、任务1.151×、冷启动1.091× | 已完成；样本很少，需扩大质量与任务覆盖 |
| Prompt/cross-KV 缓存试验 | 旧方案在短/中/长完整chunk反而慢17.33%/23.79%/6.48% | 不采用。该试验是Transformer内cross-KV复用，不是跨episode缓存T5最终embedding |
| Transformer draft 训练 | 完成18组初筛、root/中σ加权、旧root保留蒸馏、历史数据混合、夜间多轮三seed及Long补数训练；TF对容量近似MLP的部分离线配对更好 | 没有配置稳定通过原多seed准入，不能以某一seed或平均loss替换A_off；不同批次含控制组重复，不累加成独立样本数 |
| 接受器/路径对照 | 同目标DP与穷举在2920棵树等价；greedy开发结果未证明每chunk节省；严格零误差在3906 TRAIN查询零通过；MSE/RES未过双参照闭环门 | 保留当前接受器；这些负结果缩小搜索范围 |
| PF2/PF3/PF4 | 同阈值两个复用开发配对均成功，但target调用比连续OLD多13.38%/7.52%/4.64% | 没有优于当前方案的证据，不因周期更大直接升级 |
| MegaTransformer/verifier | 6份真实profile完成；最小四处pointwise融合曾过64数值案例及6完整chunk数值检查 | 用户要求回退后已撤出当前路径；0新正式速度收益，完整MegaTransformer未实现 |
| Flash-LIBERO | 审查GPU物理、任务/初态来源、接口与渲染；旧130 BDDL和20 Long初态blob来源匹配 | 只完成静态审查，未证明本项目物理/像素/成功率等价或提速，不替换当前benchmark |
| 数据与空间 | 两批target-only记录去重为493唯一初态，492完整；TRAIN194条3672chunks、CAL50条1005chunks已绑定目录和连续帧；删除授权旧快照/缓存/未完成下载 | tensor/same-query数值审计待做。项目约184→149GiB、/rjs可用约42GiB；模型、有效教师来源、tile及HW保留 |

主要原始报告：[共享verifier](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)、[O1/O2/O3](OPTIMIZED-E2E-O4-RESULT-20261013.md)、[bonus](BONUS-E2E-RESULT-20261013.md)、[夜间接受器与训练](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[早间Long10补数](MORNING-LONG10-123-RESULT-20261015.md)、[来源与清理](LOCAL-AUTODL-CLEANUP-AND-NEXT-PLAN-20261015.md)。

现有 TF64 的确是 Transformer：16个动作位置token，每位置17维，width64、4头、单层self-attention、双候选方向头、53,070参数。它没有prompt cross-attention。下一候选可用冻结T5 embedding的少量池化/降维token做轻量cross-attention，但参数量、推理开销及闭环效果都要重新比较；目前没有这样的已训权重。

## 5. 当前本机与 AutoDL 在做什么

| 资源 | 当前工作与状态 | 目的、原因 |
| --- | --- | --- |
| 本机 A100-SXM4-40G | 正式单task、多task速度完成、自有GPU已退出；新增CPU T5 bank与缓存/裁padding对照也已结束，见第9节；无新GPU实验 | 保证论文速度来自本机独占、统一硬件；不与AutoDL墙钟混表 |
| AutoDL A100-PCIe-40G | `/root/autodl-tmp/libero-scale/heldout50pair-job`，13:35启动；15:56完成22/100条、正在task2/init26的T准备阶段。计划Long10每task五初态，T/A_off共50对100条完整轨迹 | 得到跨task成败配对、环境步数和长尾，判断当前候选是否严重退化；不是训练TF，也不提供正式性能表 |
| AutoDL旧train194 | 完成10条后因NVML查询15秒超时失败，旧证据保留；现源码连续三次超时才判失败 | 原目的是A_off候选访问状态上的同查询teacher；本轮一天租期不足以同时完成新100条质量和194条完整采集，未重启该链 |

AutoDL当前输出上限8GiB，磁盘floor8GiB，快照可用约18.1GiB；GPU硬上限60000秒（约16.7小时），没有叠加新训练。按已观测吞吐保守预计10/16凌晨02:00–06:00见完整结果；若后续长horizon/失败更多、触发时间上限，应报告完整已结束轨迹和未完成项，不能把部分数据标为全量完成。

50对是描述性质量筛查，历史target元数据曾暴露，不称严格盲测。即使50对零次“T成功而A_off失败”，该单向损失率一侧95%二项上界仍约5.82%；理想零损失情形也约需99对才压至3%以下，出现损失需更多，逐task仅五对更不能保证。最终还需task分层的配对成功率净差区间。

## 6. 采集和测试是否太慢：已量化，确实需要改

读取前12条已经结束的AutoDL元数据，按 reservation→FINISH_READY（含加载、client准备、任务和drain，不含后续cleanup）统计：

| 臂 | 已结束条数 | 任务运行均值 | 完整生命周期均值 | 生命周期中非episode部分 |
| --- | ---: | ---: | ---: | ---: |
| T | 6 | 390.6秒 | 423.2秒 | 32.6秒 |
| A_off | 6 | 317.5秒 | 354.0秒 | 36.5秒 |

两臂均衡均值约388.6秒，即**6.48分钟/条**。按该局部吞吐，100条约10.8小时、1000条约107.9小时（4.5天单卡），194条A_off采集约19.1小时；这还是主要task0、包含一次T失败的前12条，不能保证后续任务同速。它用于租期和吞吐估算，**不是AutoDL与本机的硬件性能比较**。[远端元数据快照](../../../review/OMISSIONS.md)、[吞吐复算](../../../review/OMISSIONS.md)

**15:28直接读取远端请求时间的追加分析：** 15:27已完成17条（T9、A_off8），对每条读取原client的reset/policy/cache wall以及reservation→FINISH_READY时间，得到：

| 部分 | 每条均值 | 占生命周期 |
| --- | ---: | ---: |
| 模型reset请求 | 233.0秒 | 60.6% |
| policy请求 | 95.8秒 | 24.9% |
| cache请求 | 7.1秒 | 1.9% |
| 环境/本地reset/客户端余项 | 15.2秒 | 4.0% |
| 任务外启动、client准备、drain | 33.2秒 | 8.6% |
| 合计 | 384.3秒（6分24秒） | 100% |

单看已完成的A_off8条，reset平均230.7秒、policy66.8秒、生命周期353.4秒，reset占65.3%。所以即便policy继续更快，当前远端评测也被重复reset大幅限制。按17条吞吐，100条约10.7小时；其中仅reset就约6.5小时。样本主要task0/1且含一次T失败，两臂条数不等，**这些是实验周转账，不能用T/A_off的均值比当配对性能或成功率结论**。

代码已经确认CPU T5与每回合重新encode，但尚未在reset内分别计时文本编码、KV/VAE和同步，不能说233秒全部为T5。远端2线程配置与本机配置不同，也没有同硬件同负载线程消融，不能唯一归因于线程数或GPU型号。下一工程步骤应先拆reset子项并预计算固定指令embedding，再用常驻服务去掉可避免的重复加载；当下100条job保持原计划。[17条原始快照](../../../review/OMISSIONS.md)、[阶段分析](../../../review/OMISSIONS.md)

训练小draft本身相对便宜，之前多组CPU训练是秒至分钟量级；反复模型加载、文本编码、完整闭环、教师生成和实验中断才是周转大头。下一轮不能让每个loss/阈值组合都重新采一遍完整环境。

### 按优先级提速

1. **常驻一个模型，按episode完整重置状态。** 当前scale runner每条episode启动并回收server，且有单GPU锁；串行是目前调度实现，不是已经证明硬件无法并发。保留模型/权重，独立重置KV、VAE、controller、RNG、计数器和存档目录，能避免已观测的每条约33–36秒非episode生命周期中的一部分。按前12条，这部分约占9%；仅常驻服务不足以获得数量级提速。T/A_off均应用相同方式，并以原正式完整任务核验输出、动作、终态和跨episode泄漏，不另做smoke。

2. **预计算并跨episode复用最终T5 embedding，同时测reset内部子项。** Long10只有10条任务指令，负提示为空；本次100条若每次正/负均编码，需要200次文本编码，而10正+1负的固定bank只需11次，BF16 `[1,512,4096]` 共约44MiB。减少的是编码调用次数，不是已证明节省94.5%总时间。缓存key绑定文本/预处理、tokenizer、T5权重版本、长度和dtype；预计算只用公开任务指令。基线和候选同时启用；清理回合KV与状态依旧完整执行。reset内部尚未分摊前，不承诺具体倍数。建议先做这一工程改进，再追加上百局。

3. **训练数据只采一次，先离线淘汰配置。** 审核现有TRAIN194/3672chunks的tensor、prompt/KV/σ/噪声与teacher来源；可复原的target-only数据用于预训练，再补当前候选实际访问状态的same-query标签。保留验证/测试划分，不能直接把旧493条全部混入训练。官方示范也是合法数据来源，但纯动作BC不能直接监督本项目每个flow query的target方向。先离线做三seed、root保真和真实周期成本，再让少量已登记的完整DEV任务筛选，冻结后一次做标准质量表；不缩短horizon或删除失败。

4. **优先跨GPU独立分task，其次研究单卡并发。** 多GPU每卡一模型、一独立环境和会话，最容易保留现有语义；2/4卡最多约2/4倍吞吐是理想上界，不是实测。单卡40G目前占用约14–15GiB，不能仅凭这点断言第二worker绝对放不下；还须核验激活峰值、CPU内存、CPU线程和aggregate episodes/hour。同卡多模型可能只争抢算力，不能假定2倍。单模型多环境更省权重，但现接口只有一组frame/KV/VAE会话，必须显式隔离session；直接交错发obs会污染轨迹。[LeRobot官方LingBot-VA集成](https://github.com/huggingface/lerobot/blob/main/docs/source/lingbot_va.mdx)也将streaming eval限定为单环境。当前job不在中途改调度、batch或后端。

5. **reset线程、批量和算子优化依据profile推进。** 当前AutoDL OMP/MKL/OpenBLAS为2线程，T5在CPU；应测实际CPU编码阶段和4/8/16线程的生产指令bank生成成本，避免盲目堆GPU或CPU线程。固定shape的verifier可研究CUDA Graph/compile减少Python/launch开销，再选MegaTransformer原型，但可变KV、缓存写入和数据依赖接受路径需要分桶与数值检查。PyTorch2.9官方说明Graph主要减少CPU提交开销、要求固定kernel/参数地址等；它不是自动适配全部动态路径的保证。[CUDA Graph文档](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)

6. **Flash-LIBERO排在上述项目之后。** 最新仓库已有同task多world的`GpuBatchEnv`，但README明确图像渲染与goal判定仍在host边界，GPU physics不会加速CPU T5或WAM推理。按本机已测仿真step仅4.16%，即便这部分完全归零、其他不变，任务也最多约1.043×。这不是Flash实测上限，不能外推其batch吞吐；只是说明当前单环境路径的优化重点不在physics。换后端还须核验真实动力学、像素、成功谓词及配对质量，旧静态审查不是该证明。[Flash-LIBERO当前README](https://github.com/Catigeart/Flash-LIBERO)

当前full trace、teacher、profiler在质量job关闭；计数器没有逐target-forward文件I/O，只在请求结束checkpoint一次。普通输出仍异步保存，未profile前不能把日志/磁盘列为主因，也不能承诺“关日志就数倍”。正式速度入口已经使用RAM计数和episode结束存档。[quality Runtime](../../../投机采样/experiments/speculative_e2e_v3/runtime.py)、[formal RAM accounting](../../../投机采样/experiments/formal_e2e_speed_v1/server_entry.py)

资源建议：先改上述重复工作，再决定租卡。最快少改环境的扩容是增加独立A100卡按task分片；80G有利于多会话/worker内存余量，但不能保证延迟比40G快多少。H800可能加速GPU部分，却不会直接缩短CPU文本编码；当前共享kernel仍有SM80/108SM绑定，未经适配和数值验收不把H800当即插即用方案。[当前资源约束](TRAJECTORY-AUTODL-PAPER-20261014.md)

## 7. 文献对训练、接受与复现实验的实际启示

- **LingBot-VA原论文**明确用LIBERO每任务50条示范做post-training，然后每suite做3seed×500次闭环评估。训练和测试是两件事；我们训练draft也不能用训练误差代替成功率。官方仓库有Long与RoboTwin权重，第二模拟器优先RoboTwin，未见RoboDojo公开权重入口。[原论文](https://arxiv.org/pdf/2601.21998)、[官方仓库](https://github.com/robbyant/lingbot-va)
- **KERV，DAC2026**冻结OpenVLA，从LIBERO示范离线生成hidden state和action-token教师样本，再训练单层drafter；系统延迟与任务成功率分别验证。这支持减少重复仿真采集。它的离散token和运动学阈值不能直接变成我们WAM方向误差的数值阈值；Long静态阈值9→15/20的成功率49.2%→36.2%/25.8%也是放宽阈值风险的实证。[论文](https://arxiv.org/pdf/2603.01581)、[训练说明](https://github.com/zhengzihaoPKU/KERV/blob/master/docs/TRAINING.md)
- **Spec-VLA，EMNLP2025**用relaxed acceptance获得更长接受token段，但它的解码长度不等于环境执行轨迹长度；可以复现规则，不能照搬其Long阈值或速度口径。[正式论文](https://aclanthology.org/2025.emnlp-main.1367/)
- **FlowCast，ICLR2026**通过连续flow速度外推与MSE门做无训练投机，分析生成轨迹偏差。它适合作为低成本无参数draft对照，实验是图像/视频，不能据此保证机器人成功或环境步数。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)
- **SP-VLA，ICLR2026**已按动作阶段在轻量生成器与完整VLA之间调度。这说明关键动作target回退有相关先例，但该想法本身不构成本项目新贡献。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4072543747a14bbed76284cf2c04b9e9-Abstract-Conference.html)

论文应把“trace长度”拆成flow更新次数、policy chunk/target调用、机器人环境动作步数。当前每chunk仍50个flow更新；需要保持的是任务成功与环境步数分布合理，不是逐条环境轨迹严格相等。接受率与局部L2用来解释结果，不能代替成功/失败配对和任务耗时。

## 8. 剩余工作、并行关系与倒排

| 时间/工作量（北京时间） | 交付 | 依赖与判定 |
| --- | --- | --- |
| 10/15–16，当前job剩约半天量级 | AutoDL50对逐task成败四格表、双方成功步数比P50/P90、失败horizon惩罚、accepted depth/target调用 | 完整终态后统计；不是3pp非劣证明。不得将AutoDL耗时当本机速度 |
| 下一轮优先，约0.5–1工作日 | 常驻服务、reset内部计时、T5 embedding bank及原完整任务等价核验 | 先解决重复工作；基线和候选同步改，另开新job，不改正在运行的源码/计划 |
| 可并行，约0.5–1工作日审计，teacher生成另看实测 | 旧TRAIN tensor/same-query数值审核，固定FIT/CAL；预训练数据与on-policy标签明确分开 | 元数据审核已完成，不能把它当tensor/同查询通过。缺少的标签批量生成或定向采集 |
| 10/17–20，数据合格后约1–2天研究周转 | 相同数据/预算的MLP、TF64、prompt条件TF三seed，CAL选结构/阈值，完整DEV质量 | 训练本身便宜，数据和闭环耗时决定交付；未过则继续A_off。不能现在承诺最终TF已定 |
| 10/20–26 | 有限接受器消融；并行隔离verifier profile/一个MegaTransformer原型 | 算子线以冻结A_off作为输入基线；先数值再成本，原tile不改，不靠融合覆盖算法质量问题 |
| 10/24–31 | 冻结后Long10扩大质量与论文速度四层表；条件允许做RoboTwin | 先用50对估计不一致率，决定250对或更大样本；标准500对描述性表与留出共享轨迹须说明，不重复运行 |
| 11/01 18:00 | 给老师完整中文稿与已有主表 | 若10/22前仍没有跨task质量证据，缩小算法主张和实验范围；不把未测Mega收益当主结果 |

中文方法与实验协议现在可在会话逐段写，用户整理到飞书；不生成本地论文初稿。论文需要方法、系统、质量、环境步数及单chunk/自然任务/冷启动等速度表；当前缺少最终TF、稳定质量和MegaTransformer净收益，不能称全文实验已齐。

扩展范围也需收敛：Long10质量链完成后优先第二模拟器RoboTwin；LIBERO-100、RoboDojo、真机和FastWAM/FlashWAM目前均没有本项目实测，不同时作为这轮必须完成项。真机不是所有推理加速论文的硬性前提，KERV也明确未做真机适配，但这不保证CVPR审稿接受我们的实验范围；跨任务、第二模拟器或第二模型以及清楚的系统贡献可以增强证据。KERV已有动态阈值、SP-VLA已有动作阶段调度、FlowCast已有连续flow投机，因此换小TF或加一个阈值本身不足以作为新贡献。应围绕WAM候选路径的质量控制、同查询训练和共享verifier的净收益建立主张，未验证部分保留为计划。

[CVPR2027官方时间](https://cvpr.thecvf.com/Conferences/2027/Dates)：注册11/10 AoE，即北京11/11 19:59；正文11/16 AoE，即北京11/17 19:59；补充材料11/23 AoE，即北京11/24 19:59。官网未单列abstract截止，内部按注册日锁定标题/摘要/作者，并在11/08前完成注册检查。11/01交老师的内部线继续保留。

本轮执行清单：

- [x] 统一自然任务与冷启动E2E口径，独立CPU复算闭合耗时账。
- [x] 汇总正/负优化、结构与接受器实验，不混合不同分母。
- [x] 读取本机/AutoDL运行状态，量化串行吞吐与100/1000/194条时长。
- [x] 核查常驻、文本缓存、离线教师、并行、CUDA Graph及Flash优先级与边界。
- [x] 保存图、JSON证据并同步HW入口与Knowledge。
- [ ] AutoDL100条终态及配对质量分析（运行中）。
- [ ] 常驻/文本缓存实现与完整任务等价检查（本轮仅方案，未实测收益）。
- [ ] TRAIN数值审核、最终TF选型与独立闭环、扩大质量、MegaTransformer净收益。

## 9. 后续答疑：结构对照、PF2与其他WAM的reset

**AutoDL当前这100条不负责选MLP/Transformer或PF2。** 它仅比较冻结A_off MLP连续投机与T，提供当前算法的跨task质量参照；新的194条same-query采集曾失败，后续TF全量训练链没有完成。它不能直接回答最终draft选型与训练方案。

MLP/TF离线对照已经执行：初筛同3906 TRAIN/1890 DEV、两个学习率、三个seed、各1000更新。容量匹配MLP96为52,928参数、TF64为53,070参数；lr1e-4三seed中位root代理通过数分别476/630与515/630；TF在同seed/学习率的6/6配对更好。历史A_off参照是527/630，但预训练历史不同，不作为完全公平的scratch结构比较。扩充数据后TF三个seed为530/532/528，匹配MLP为510/512/517；TF仍在task0和旧通过节点保留方面未达冻结准入。**离线门未过不等于TF的LIBERO成功率已经失败。完整TF/MLP配对闭环至今没有完成，不能说MLP最终优于TF。** r7/r8等中断主要属于入口和资源问题，不能记作TF质量失败。[18组对照](DRAFT-ARCHITECTURE-RESULT-20261014.md)、[最新三seed与闭环边界](TF-DRAFT-STRUCTURE-AND-QUALITY-20261015.md)

本项目PF2确为chunk序列`投机、投机、纯T`反复，代码按`ordinal % 3 == 2`选择target refresh。K2/B2是**同一个chunk内**两层、每层两个方向候选，bonus是消费已算出的叶target方向；均不等于PF2。当前AutoDL没有打开周期PF2。[PF2审计源码](../../../review/OMISSIONS.md)

FastWAM方面，今天重新核查了官方commit `7faa71108368fbb3b6885649f112af607427a2d4`：其LIBERO manager采用每GPU一个常驻模型worker；模型的`.to(device)`也移动text encoder，文本编码不必像我们当前设置一样固定在CPU。`infer_action`收到prompt时仍执行文本编码，也支持直接传预计算context/context_mask；官方训练脚本另有text embedding预计算。不能把“训练支持cache”写成“评测自动缓存文本”。这些源码说明每回合重新加载和分钟级CPU文本编码不是所有WAM必然具有的成本，但**没有本机同口径FastWAM reset实测，不能给它断言零开销或具体秒数**。[manager](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/run_libero_manager.py)、[text/infer接口](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py)、[来源清单](../../../review/OMISSIONS.md)

按用户先定位/优化reset的指令，本机完成独立CPU产物任务[reset_text_bank_v1](../../../投机采样/experiments/reset_text_bank_v1/README.md)：原T5/tokenizer与原text helper计算十条公开任务指令和空负提示，共11个BF16/512token embedding；七CPU线程、CUDA不可见，最多30分钟。首job在读取已知task8/init42失败元数据时因其没有prompt字段报错，0模型加载/0编码/0GPU，原失败和源码保留；修正为仅从完整记录提取公开指令后另建r2，没有删除或掩盖失败。

**r2已完成11/11，CPU产物与缓存接口验证通过：** 全程716.53秒；纯编码706.01秒，其中T5 forward705.81秒（99.9717%），分词及输出整理合计约0.20秒。每条原512token编码62.02–67.10秒，生成bank为46,142,210字节（约44MiB）。全部11个实际embedding保存重载、缓存返回值逐位一致，返回存储独立，修改返回值不污染bank，未知prompt/错误dtype/长度/batch拒绝，CPU RNG不变。原始模型未加载、0GPU、0LIBERO episode，两个自有CPU进程均正常退出。这里“原始模型未加载”指target transformer，T5已实际加载执行。[完整CPU产物](../../../投机采样/results/runtime/p0-reset-text-bank-v1-20261015-cpu7-r2/RESULT.json)、[缓存CPU验收](../../../投机采样/results/runtime/p0-reset-text-bank-v1-20261015-cpu7-r2/CACHE-CPU-CHECK.json)

还完成一个有明确负结果的变更实验：保持原文本helper与11条真实输入，将进入T5的已mask padding截到有效token后再补回512。编码合计69.38秒，但**0/11与原BF16输出逐位一致，最大绝对差0.16015625**。保留`FAIL_BITWISE_PADDING_PARITY`，不将它接入冻结target，也不把69秒包装成已验收reset/E2E收益；尚未定位数值差异的唯一原因或测闭环影响。[裁padding完整结果](../../../投机采样/results/runtime/p0-reset-trimmed-text-v1-20261015-cpu7/RESULT.json)

这些是本机CPU文本阶段的直接测量，不能直接归因AutoDL全部233秒reset。默认关闭的bank adapter已写好，尚未接入当前AutoDL或在线server；GPU传输和完整任务等价验收仍需完成，不能宣布新的reset/E2E加速。下一步先完成共同用于T/A_off/TF的缓存reset等价验证，再以相同数据/接受器完成TF与容量匹配MLP的完整闭环，PF2另作单变量。15:56 AutoDL为22/100，仍是T/A_off质量对照，没有开始TF或PF2筛选。
