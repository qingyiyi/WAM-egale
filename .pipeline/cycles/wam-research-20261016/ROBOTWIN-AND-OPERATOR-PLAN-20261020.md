# 单卡RoboTwin迁移与单层算子并行计划

2026-10-20，北京时间。状态：`PLAN_READY_NOT_STARTED`。本轮交付规划；没有安装新环境、下载权重、训练、启动RoboTwin或新增算子GPU作业。下述预算是建议上限，不继承旧作业余额，不重置旧截止。

## 已完成与待执行

- [x] 核对HW、当前算子、真实训练链、官方RoboTwin接口与远端资源。
- [x] 给出两卡分工、有限迁移面板、算子依赖顺序和时间/止损。
- [ ] 执行阶段的源码、任务seed/指令、预算和绝对截止冻结。
- [ ] RoboTwin原Target完整任务复现、投机适配、数据/训练与三臂DEV。
- [ ] 算子数值修复、资源瓶颈优化、跨阶段tile就绪调度及强基线比较。

本计划的资源意图见[用户决定](../../memory/global/requirements/decision-单卡规划RoboTwin迁移并行推进算子.md)。历史M17/M35/M39结果和四项false保持；计划完成不等于上述待执行工作完成。

## 1. 动态事实与资源分工

[12:03:15北京只读快照](../../experiments/robotwin-migration-survey-20261019/RESOURCE-SNAPSHOT-20261020.json)：远端两张A100-SXM4-40GB均0MiB、0%且无compute进程；driver535.161.08。~/cxz所在文件系统空闲约12.62TiB，这是共享文件系统可见余量，不是本项目专属额度。cgroup配额45个逻辑CPU等效，不能将可见256CPU当作可用配额。

现有va环境Torch2.9.0+cu126，未发现sapien/mplib/gymnasium/vulkan的Python发行包；仅查了该环境元数据和三处一级目录，不声称已全面确认系统Vulkan或其他环境不存在。

| 资源 | 建议用途 | 隔离与并行 |
|---|---|---|
| GPU0 | 原真实LIBERO单层算子，保持先前测量GPU UUID | 平时编译/正确性可与GPU1并行；正式计时避开另一线重CPU/IO阶段，记录整机负载 |
| GPU1 | RoboTwin Target、采集、部署接口校准与T/C/P完整DEV | 初始一模型worker；官方offload约24GB，叠加B14/模拟器峰值未知，不能假设40GB能放两worker |
| CPU | 算子2核编译/测试，RoboTwin server/client各2核、控制1核，训练另2核 | 初始总计不超过9核；不重叠绑定，启动时按实际拓扑确认，CPU不足不以无限并发补偿 |
| 磁盘 | 所有新目录、环境、HOME类缓存重定向及临时文件限~/cxz | 模型/环境/资产建议100GiB软上限，确认资产清单后才下载；新采集/审计≤12GiB，少量重放fixture包含在内≤3GiB |

新环境建议~/cxz/robotwin/{source,envs,assets,models,runs}；这是拟议路径，目前未创建。不修改现有va/libero环境，不升级驱动或CUDA13，不连接已归还AutoDL。RoboTwin使用SAPIEN/Vulkan；不能将LIBERO的EGL配置机械套用。若Vulkan缺系统能力，报告具体缺项，不擅改系统。

两卡并行适合质量和开发，不自动构成严格性能隔离。正式计时仍需独立进程重复、交错顺序及主机负载审计。

## 2. RoboTwin：先证明迁移成立，再决定规模

### 2.1 固定来源与必须适配的接口

官方[Target权重](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)可用，revision为`8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`。沿[固定LingBot-VA README](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/README.md#evaluation-on-robotwin-20)固定源码`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`，RoboTwin固定`2eeec322d95799f537cbfe5f291a8220d965ccb8`；不从main替换兼容实现。SAPIEN3.0.0b1、mplib0.2.1和三相机渲染先核实。模型无需重新大训练。

| 项目 | 迁移方案 |
|---|---|
| 条件/动作 | 三相机，16有效动作通道、30维padded teacher；保留EEF偏移、四元数归一化、gripper和原命令映射 |
| draft | TF64主体仍1层、hidden64、4heads、FFN256；16→32token，17→35输入特征/位置，双7维头→双16维头；位置编码/输入输出层与normalizer重建，LIBERO权重不直接加载 |
| RNG与latent | 保留teacher原[1,30,2,16,1]的960元素噪声分配及inactive mask；draft有效表示512元素不代表可减少RNG抽样 |
| scheduler | 官方video25/action50及SNR shift1.0；不沿用LIBERO的20/50和0.05 |
| verifier | 保留K2/B2、7状态、CFG/bonus/prefix_then_cost方法；审计B2/B14同query差异、共享历史/私有KV、末尾调用与实际g |
| PF2 | S S T，每个完整policy请求成功返回才计数；T走canonical完整action target，不是bonus，不重置物理状态 |
| reset | 重建该Target/任务指令的完整BF16 text bank；所有臂同用等价优化，精确指令匹配，miss/cold另列；不能直接复用LIBERO内容 |
| 实现隔离 | 新RoboTwin适配目录；不原地修改LIBERO生产配置/训练表/旧证书；暂不接入尚未数值准入的整层persistent kernel |

现有算子只验证Q16/Q128，RoboTwin静态配置为actionQ32/videoQ240、更长pool9792；后者尚未现场捕获。因此两线先并行开发，不能边迁移环境边更换未验证kernel，让质量差异失去归因。

### 2.2 默认面板与数据链

仅`demo_clean`。先固定6任务：`place_object_basket`、`stack_blocks_two`、`handover_block`、`lift_pot`、`press_stapler`、`open_microwave`，覆盖摆放/堆叠/交接/双臂抬物/按压/开合。它们来自已核验的[50任务名单](../../experiments/robotwin-migration-survey-20261019/TASKS.csv)，分组是工程抽样，不是官方suite。环境不支持某项则BLOCKED，不观察成功率后换容易任务。

每task预先通过官方专家可解性筛选形成**19个有效seed**：8个Target数据、3个parent访问数据、3个current访问数据、5个最终DEV。记录所有被跳过seed及原因；筛选与策略成功无关。先冻结seed、指令、物理/控制器/相机与RNG清单，再执行策略。19组是各阶段隔离初态，不是同一seed换文件名。

| 阶段 | 完整episode数 | FIT/CAL与用途 |
|---|---:|---|
| Target数据 | 6×8=48 | 每任务FIT6/CAL2，总36/12；首批每task1局即6局属于已划定FIT，用于原Target复现和估时 |
| parent访问数据 | 6×3=18 | 每任务FIT2/CAL1，总12/6；生成者为本环境PARENT2000 |
| current访问数据 | 6×3=18 | 每任务FIT2/CAL1，总12/6；生成者为本环境CONT3000 |
| 冻结后的DEV | 6×5×3臂=90 | 30个配对初态，T/TF/TF+PF2各30；全部新跑，独立于上述采集初态 |
| 合计 | **174局** | 84采集＋90 DEV；只是有限迁移pilot，不是50任务标准评测或独立TEST |

失败的完整episode保留并按预登记规则取实际访问query。每局自然跑到成功或原任务step_lim；基础设施中断另列PARTIAL，不伪造失败、不自动补跑。只有实际到达/查询身份明确的监督进入相应训练池；整棵候选树不冒充机器人实际轨迹。保留少量完整条件fixture，其余按共享引用存储，避免每query复制30层KV。

### 2.3 训练：复用方法，不把旧权重当新环境模型

现有三小套件实际链是PARENT2000→CHILD2500，再从CHILD2500分别做CONT3000/ON3000；ON数据由CONT生成，不是ON自身on-policy。来源：[training.py](../../../投机采样/experiments/tf_suite_pipeline_v1/training.py)、[adapter.py](../../../投机采样/experiments/tf_suite_pipeline_v1/adapter.py)。

迁移默认复用这一链和损失，缩小**采集规模**而非声称复现全量LIBERO训练：seed20260923，batch512，AdamW/weight_decay0.01；起步2000updates/lr1e-4，后续每段500/lr3e-5；后续target/访问态FIT各50%。总计算3500updates，最终ON checkpoint的谱系3000updates；不把总训练工作量说成3000。normalizer仅Target FIT拟合一次，之后冻结。只评估固定最终ON，不按CAL/DEV挑checkpoint、seed或架构；CONT保存为来源/后续对照，不加机器人实验臂。

loss仍回归decoded raw velocity；gap项才按residual_std归一化。先核对BF16部署decode与训练/离线接口，以及teacher CFG后标签与query对应。训练列必须适配到512/1027维等真实新接口，不能绕过旧112/227断言把数据硬塞进去。

接受公式仍为`abs(Δσ) * ||v_draft-v_target||₂`，但LIBERO数值tau没有跨动作表示保证。建议在RoboTwin独立Target CAL12上，用固定PARENT2000最小头局部误差的**任务/episode等权90%分位数**建立一次工程初值；加权分位数定义为加权经验CDF逆，平局取最小值，运行前冻结实现。这与旧单task/混合边线性q90并非同一统计协议，必须另列新tau身份和量纲。之后parent/current采集、最终ON、C/P都不再调tau，额外CAL仅审计漂移。q90不保证成功率、也不保证90%接受；DEV失败后不按task改阈值。该校准是本计划建议，尚未计算或批准执行。

若数据分布/部署接口审计失败，停止该链并报告；不在本轮追加视觉cross-attention、架构/seed搜索或第二轮数据循环。

### 2.4 质量、成本与进退

三臂轮换顺序、同case同卡；T/C/P初态严格一致，C/P前两S严格审计，分叉后允许真实状态自然不同。首两S失败不能被总体成功数覆盖。

逐task及全体报告SR、T胜C/P败与反向胜出、完整失败原因；共同成功的命令长度比P50/P90/max（NumPy linear），全体实际命令数与失败罚分分列。RoboTwin按任务原上限H_i计失败，不套800；不同任务汇总同时给任务等权归一化罚分。完整T失败也保留，不能筛到“Target成功”才计入DEV。

成本记录action、完整policy、reset/cache、环境、冷setup的互斥边界；核心`ΣT_round/Σg`，加上完整chunk和全attempt自然episode成本、成功吞吐。共享CPU下计时只作探索；冻结fixture的独占replay回答固定计算成本，**不能重放hidden就称测到机器人E2E**。正式E2E仍须重新完整运行环境且获得相应质量/性能许可。

建议进入下一预算讨论的筛查：30配对全部有效，无T成功/候选失败；候选SR不低于T；共同成功长度P50≤1.10、P90≤1.20、max≤1.30；全体归一化罚分不恶化；审计完整且成本有收益信号。这是保守pilot投资线，不是旧门修改或30局非劣证明。C/P分别判断，PF2不预设更好。失败或成本UNKNOWN则停在迁移DEV，不自动扩50task；原四false不变。

通过也只能说“在6任务迁移DEV有信号”。官方50任务×100有效seed×三臂=15000局/条件，与本面板174局完全不同；全量、Hard、少步数Target和MLP论文对照均需另排预算。若论文声称优于少步数或TF结构更好，不能省对应对照；首轮不承担这些结论。

## 3. 算子：先把正确性与单tile算力做好，再调度依赖

当前证据：[FA2-RESULT](../../experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)。self/cross同输入18项逐位PASS，但整层9项FAIL，相对L2约0.133%–0.563%；完整单层仍慢native3.20–7.60倍。108CTA×8warp，255 registers/thread、82464B shared、2168B local；NCU流量/实测occupancy UNKNOWN。这些不是新一轮结果。

### O1：先解决明确定位的数值边界

优先长KV video FFN2：该case直到FFN1/GELU已逐位一致，最适合排除上游误差。随后B14/B2 Q投影、短中video Norm2的12/8个差异。对同一冻结输入核对cuBLASLt实际选择、K归约/累加分区、BF16舍入、bias/affine和编译指令；不能仅照抄Torch的Python Linear调用就声称得到cuBLASLt设备实现。

旧匹配split/tile几何的CUTLASS候选四项已失败，不重复无依据大网格；保留反证。每个改动先独立同输入边界，再回整层，保留FA2数值路径。目标是原严格契约，不能以0.5%相对误差替代逐位门。

时间建议0.5–1.5工作日、GPU生命周期≤2h、≤300完整block调用、新增证据≤1GiB。若供应商内部GEMM路径不可复现，给出具体阻塞及数值差异，保留独立原型；不承诺一天内一定逐位对齐，更不悄悄改门。

### O2：完整单层恢复验收，并先优化占比高的原语

九签名B2 action/B14 action/B2 video×短中长，核对最终hidden、输入/参数/RNG、所有有效KV及合法更新顺序。KV比较必须直接对原Torch；不能只沿用“相对旧async通过”。精度与性能使用同一输入manifest，profile与计时分开。

性能先处理GEMM和整层寄存器活跃期、shared布局、加载/计算重叠；FA2短Q补零和split/combine另分解。先量出各阶段time与等待，区分编译资源数和实测spill/occupancy；权限缺失保留UNKNOWN。单次只动一个有profile依据的tile/流水因素。更大tile可能减少CTA并增加资源，不能假设一定更快。

建议1工作日、GPU≤2h、≤400完整block调用、≤1GiB。目标先达到强baseline附近（工程筛查≤5%慢，另报独立进程波动），此数不是放宽精度，也不是新增收益。强baseline包括现有适用cache/Graph；共同缓存收益另列。

### O3：整个单层内的tile依赖与warp流水

保持一个kernel。先保留已对齐数学/归约边界，把全局“整阶段结束”改成明确tile就绪；不是将self/cross注意力移出kernel。实际DAG包括：

`Norm/调制 → Q/K/V+RoPE → self-attn → 输出投影/残差 → Norm2 → cross-attn及投影/残差 → Norm3/调制 → FFN1/GELU → FFN2/残差`。

条件K/V可提前准备但需保持公共强baseline一致；不把缓存本身包装成MegaTransformer增量。具体阶段以whole_layer.cu为准，不用图省略真正归约。

- Norm等待一整行隐藏维归约；attention等待该query合法可见的K/V，保留history/private边界。
- self/cross输出投影需对应所有head输入；FFN2需要完整F14336归约输入，不能把一个FFN1 tile生成当作FFN2全输入就绪。
- 初版优先选**FFN1/GELU→FFN2**单个交接：第一层全部输出tile对一个row组到齐后，即可让该row组的FFN2开始，无需等所有row组。完整K归约次序仍不变；这是跨阶段重叠，不是把不完整归约当结果。
- attention仍完整tile化；其4计算warp现已承担load/MMA/softmax/store，后续独立评估把加载交给专用warp的可行性。不能在未修改其barrier/数据生命周期前直接将FA2载入拆走。
- warp0调度、loader/compute/store分工沿现有实现；就绪计数/epoch、缓冲引用计数、release/acquire与workspace回收需明确。避免所有驻留CTA自旋等尚未调度producer导致死锁。

先一条依赖交接，再拓展self/cross阶段，不一开始替换全部barrier。SM80使用现有cp.async和显式同步，不引入Hopper TMA/WGMMA假设。

建议1–2工作日、GPU≤3h、≤500完整block调用、≤1GiB。O1未通过时只能保持独立研究原型并分析结构，不能接入机器人或签发精度PASS。

### O4：证明新增调度收益

同一真实输入的强baseline与新kernel，候选数1/3/7分别短中长KV；3状态须来自合法真实子集，不伪造新的生产树。现有缓存与新调度分开做2×2：调度off/on × 缓存off/on，双方合理配置。单状态不能故意弱baseline。

比较单层中位数、尾部和独立进程差异；若以后集成verifier，才统计真实g与`ΣT_round/Σg`，不能从单blockreplay虚构提交数或机器人速度。

建议0.5工作日、GPU≤1h、≤300完整block调用、≤1GiB。O1–O4初始累计≤8GPU生命周期h、≤1500block、≤4GiB；每次再冻结实际call清单/进程deadline，沿原M39记录、不新增研究注册目录、不改旧预算。到预算或10/22检查点先交结论，不能因追求“一定更快”无限延长。

若同input精度未过、速度仍显著落后，10/22明确NO-GO并收缩本轮系统claim；baseline仍是可运行对照，optimized研究保留。只做一层，不扩成30层大kernel。

## 4. 墙钟、GPU预算与关键路径

以下从**实际执行T0**起估算，不表示今日已启动。

| 工作 | 常规工程墙钟 | 初始GPU生命周期建议帽 | 主要不确定项 |
|---|---|---:|---|
| RoboTwin版本/环境/原Target首6局 | 0.5–1.5工作日 | 2h（含在下行总额） | Vulkan、资产下载、内存与完整episode长尾 |
| RoboTwin接口/采集/固定训练链/90局DEV | 再1.5–3工作日 | 总≤24h：准备/完整调用审计2h、84局采集8h、训练/CAL1h、DEV12h、清理余量1h | 实测单局和专家筛选速度；分项不能借额突破总帽 |
| 算子O1/O2 | 1.5–2.5工作日 | ≤4h | cuBLASLt数值复现，不保证成功 |
| 算子O3/O4（前置满足才实施） | 再1.5–2.5工作日 | ≤4h | 原语吞吐、依赖并行度、寄存器压力 |

RoboTwin算子式首调用校验限≤24完整policy请求、≤1500 teacher尝试，包含在准备2h；这是完整调用正确性，不是截断episode的成功率测试。机器人每局建议≤1800s、≤8000 teacher尝试，174局总尝试≤1393500（含前检1500）；任一帽先到即停止，不能延horizon冒充完整。开局须预留整局上限与至少15min清理，总24h不足则报告PARTIAL。等待/失败/准备都计生命周期，CPU环境安装未持有GPU时另计墙钟。

24h是上限而非保证174局完成。仅完整episode均值假设为2/5/10min时，174局分别5.8/14.5/29GPU小时，尚未含专家筛选、训练、冷启动和审计；10min情景明显超帽。先6局得到每task耗时、long-tail、显存和储存增量，再预测P50/P90完工区间；不足时缩小后续承诺，不能自动追加授权。

| 节点（北京） | RoboTwin线 | 算子线 |
|---|---|---|
| 10/20–21（若立即进入执行） | 环境/接口与首批完整Target，给可信估时 | O1数值归因，优先干净FFN2边界 |
| 10/22 go/no-go | 至少明确Target是否可复现、迁移阻塞和剩余耗时 | 精度、原语效率与新调度投入结论；证据不足停止扩方向 |
| 10/23 | 稳定时冻结迁移API/任务清单；未完成结果标候选 | 冻结本周期引擎/claim；未通过不宣称生产替换 |
| 10/24–26 | 有预算且接口稳定时完成有限DEV/报告；不自动全50任务 | 只完成已冻结验证及获准计时，不无限改kernel |
| 10/28 | 主表；未完成第二环境则明确删减该主张 | 主表与失败/局限收口 |
| 11/01 18:00前 | 进入给老师的中文证据包 | 同左；正文仍由用户飞书整理 |

乐观：环境顺利且数值复现可解，2–3天有RoboTwin迁移信号与算子阶段结论。常规：RoboTwin3–5天，算子3–5工作日（包括不成功的结论）；两线并行，不相加。受阻：Vulkan/权重接口/数值契约卡住，10/22先报阻塞、10/28保留LIBERO主结果；不为赶日期更换协议或伪造PASS。

## 5. 交付与范围

RoboTwin最终产物：版本/资产/初态manifest、实际训练谱系与FIT/CAL/DEV清单、174局覆盖/缺失索引、三臂逐task成功/长度/失败罚分、调用/耗时账本、少量可重放fixture、资源清理和下一投入决定。

算子最终产物：原Torch对照的逐阶段/整层/有效KV误差表、相同输入性能表、warp/tile DAG与可见性约束、原语/等待归因、强baseline增量和未解决问题。baseline/现有投机源/原tile/保护HW不改。

本计划不把四项false改true，不补旧失败，不自动全量RoboTwin/Hard/新架构/第二轮训练/独立TEST/正式机器人性能。当前完成的是可执行规划；原任务没有因此恢复或新增排队。
