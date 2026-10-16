# 投机采样项目现状与收口建议

> 最新夜间结果（2026-10-15T05:45:28+08:00）：45次训练无稳定新权重，保留A_off；30条开发质量任务、新TRAIN与PF2/3/4已独审。算子融合已撤回，原tile不改。见[统一报告](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[扩大准备](NEXT-SCALE-READINESS-20261015.md)，HW记录 CHAT-20261015-OVERNIGHT-DRAFT-ACCEPTANCE-RESULT。


> **2026-10-14 研究追加（CHAT-20261014-DRAFT-DATA-VERIFIER-FOLLOWUP）**：历史Long4802行重建、15组CPU训练、6份真实profile、64个CUDA数值案例及6个完整chunk验收已完成；新draft未通过多seed原准入，保留A_off。四处融合数值一致，正式性能待独占；42chunk私有导入入口已封存。累计GPU31065.8421秒/147346 intents，余82.24分钟，自有训练/GPU已退出。详细结果与下一步见 [本轮报告](DRAFT-DATA-VERIFIER-RESULT-20261014.md)。以下原报告与历史状态保留。


> 最新结果（2026-10-14T20:10:22.540+08:00）：[18组draft结构初筛与verifier状态](DRAFT-ARCHITECTURE-RESULT-20261014.md)：Transformer在6/6组公平配对中优于容量相近MLP，但全部候选未过准入，保留A_off；低学习率root中位515/630对527/630，退化集中中sigma root。先核验旧train来源并扩大有效数据。verifier profile代码/CPU预检完成，GPU外来占用未启动，尚无融合原型或新性能收益。本轮0新GPU、余90.83分钟，自有训练/监测已结束；tile及preset保留。Knowledge：CHAT-20261014-DRAFT-ARCHITECTURE-SCREEN-RESULT。下方“尚未训练”等为历史时点。

> 最新安排（2026-10-14T19:16:40+08:00）：[draft选型与verifier并行计划](DRAFT-STRUCTURE-PARALLEL-PLAN-20261014.md)：先冻结接口，有限对照三类小候选，筛选首版结构后扩大训练分区；系统线可同时profile当前CFG14并做隔离block原型。现有训练便宜，优先补数据覆盖与闭环证据；旧Transformer shadow不能选定当前结构。已完成静态复核，0新GPU，尚未启动新训练/profile；余GPU90.83分钟。tile和当前preset不改。Knowledge：CHAT-20261014-DRAFT-STRUCTURE-PARALLEL-PLAN。

> 最新完成（2026-10-14T18:53:35.594+08:00）：[三档完整policy chunk实测](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)：短/中/长KV加速1.602×/1.544×/1.500×，45完整chunk、18/18正式配对胜出，同算法现场数值/KV/RNG及独立CPU审计通过。两次失败保留，最终累计GPU30550.488秒/36000秒（余90.83分钟），前向收费144550；自有进程清理完成。此前“待办/本次未测得”仅为当时状态，以本结果为准。[MegaTransformer整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已固定范围，下一步profile和单block原型；tile代码/HW不改，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-LENGTHS-COMPLETED。

> 最新更新（2026-10-14T18:32:43.139+08:00）：[当前完整chunk测试状态](CURRENT-CHUNK-LENGTHS-STATUS-20261014.md)与[MegaTransformer/tile整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已归档。三档入口/CPU预检完成；本机实测模型加载时外来lhw GPU进程出现，中止并清理，0完整chunk/0性能样本。累计GPU30110.927秒/36000秒，前向保守收费136964；下一新job继承本次失败账本。统一研究目标，tile原源码/HW保留，MegaTransformer仅固定首版范围，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-ATTEMPT-TILE-INTEGRATION。

> 最新安排（2026-10-14T17:40:03.895+08:00）：[总进度、待办及并行依赖](PROGRESS-AND-PARALLEL-PLAN-20261014.md)。短/中/长KV当前完整chunk实测已按用户最新指令改为待办；本机GPU正被他人使用，仅查询状态，0模型/0测试/0新增代码，不自动排队。下一测试资源已选本机独占窗口，无需重问偏好。算法质量、verifier分析设计与会话写作可并行，正式GPU计时须单独窗口；reset后置。当前CPU一核轻量整理，tile/旧质量门/preset不变。Knowledge：CHAT-20261014-SPECULATIVE-PROGRESS-PARALLEL-BENCHMARK-DEFERRED。

> 投机主线优先级更新（2026-10-14T17:28:57.196+08:00）：先完成投机质量与短/中/长KV完整chunk证据，再推进基于profile的MegaTransformer verifier；reset优化后置，有余时再做，仍计入完整任务成本。已测无reset长KV完整chunk相对T约1.48×，仅两个状态，非新PF2/总体收益。[决策与复算](RESET-DEFERRED-SPECULATIVE-PRIORITY-20261014.md)。当前本地GPU暂停、CPU一核；preset/旧质量门/tile不改。Knowledge：CHAT-20261014-SPECULATIVE-RESET-DEFERRED-PRIORITY。

更新：2026-10-14，Asia/Shanghai。当前资源约定：暂不使用GPU，CPU检查固定一个核、低优先级串行；没有排队、训练、仿真或自动续跑。tile transformer / tile-sdpa代码及其HW内容不在本轮整理范围。

**判断：工程路线可行，已有真实推理和小规模完整任务收益；稳住闭环质量的方法贡献尚未闭环，当前证据还不足以支撑成熟的CVPR主结果。** 已知局部加速值得保留，下一阶段应集中回答“在更广任务上，何时接受、何时走原target，能否同时保持成功与任务长度”。本页归纳已有实验；本轮没有新增实测性能。

**最终目标与当前实现。** 对第一代公开LingBot-VA的action去噪分支，用很小的draft预测未来方向，target批量验证候选树，接受合法前缀，拒绝时回退；在成功率不明显下降、完成任务步数和长尾受控的条件下，减少完整推理及任务墙钟。当前研究冻结video20/action50和原LIBERO执行协议；不是减少原采样步数的实验。学习型双头MLP约2.6万参数，K2/B2展开7个逻辑节点，经CFG形成14行target验证。bonus消费已算出的叶target方向，再推进一次chunk内部去噪。PF2则是跨chunk的spec/spec/完整target50更新；两者单位不同。

正式配置[当前候选](../../../投机采样/configs/current_speculative_candidate.json)仍指向prefix_acceptance_v1：A_off、bonus、共享KV、O1/O2/O3、prefix_then_cost及原WL2阈值。PF2及MSE/RES是隔离研究候选，未改默认配置。periodic_refresh_quality_v4中的v4是软件版本，实际周期仍为PF2；没有完成PF3/PF4周期优劣对照。

**已经完成的工作与收益。** 先后完成真实状态/动作/cache/RNG合同、完整LIBERO采集与checkpoint匹配；小draft训练、tree验证/合法前缀/bonus；共享KV与主机调度优化；同查询教师标签、保持训练、阈值及周期target实验；接受规则文献和Flash静态审查。独立审计、失败留档和预算清理已建立。下面不同实验各用自己的基线，收益不可直接相乘，也不能归为最新PF2配置的速度。

| 已完成项 | 已测收益 | 证据范围 |
|---|---|---|
| 共享self-attention历史KV | 完整verifier 5.171→3.360秒，耗时下降35.02%；额外峰值891→257 MiB | 固定长KV、完整30层/29query、六组交叉对照，输出逐位一致；不是整机总显存或E2E比例 |
| O1模板复用＋O2减少标量同步＋O3工作区复用 | 相对旧S0，完整两任务707.183→642.318秒，耗时下降9.17%；轨迹与动作相同 | 同一task、两个初态的小规模配对；没有跨任务稳定性保证 |
| bonus及既有工程优化 | 两个长KV完整chunk相对原T约1.48×；相对OFF约1.16–1.17× | 固定真实状态、单部署模型、每状态六轮；包含完整video/action推理 |
| 同轮bonus完整任务 | 相对T约1.21×、耗时下降17.36%；相对优化OFF约1.018×（耗时下降1.76%） | 一个task两个初态，三臂均2/2成功；其中一初态比OFF更慢 |
| 周期target PF2 | init13/14的四对条件：T/PF2均4/4成功，惩罚步数1111→1121（+0.90%），最差变长2.55% | 通过当组双参照开发质量门；下一组出现长尾，不能外推保证 |

来源：[共享verifier](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)、[O1/O2/O3完整任务](OPTIMIZED-E2E-O4-RESULT-20261013.md)、[bonus完整chunk与任务](BONUS-E2E-RESULT-20261013.md)、[训练/阈值/PF2](TRACE-SUCCESS-CONTROL-RESULT-20261014.md)。共享verifier历史父监督器退出FAIL及后续CPU证据协调均保留，表中仅引用已完成测量。

**质量问题目前仍存在。** 最近一轮固定其他条件，只比较三种接受规则；两个task×两个init，每臂4条完整任务，共16条，另有20个GPU完整chunk数值控制。失败按800步计入质量门，以下不能当作全LIBERO成功率。

| 配置 | 成功 | 惩罚步数总和 | 相对T最差变长 | 当轮结论 |
|---|---:|---:|---:|---|
| 原T | 3/4 | 1640 | — | 参照 |
| PF2＋WL2 | 4/4 | 1144 | +26.62% | 总量较好，单对长尾越门 |
| PF2＋MSE | 3/4 | 1655 | +7.98% | 对T的门通过；相对WL2丢失一条成功，双参照门未通过 |
| PF2＋RES | 4/4 | 1207 | +20.23% | 保住这四例成功，长尾仍越门 |

对应[完整报告](ACCEPTANCE-COMPARISON-RESULT-20261014.md)。MSE不能被写成“完全无效”：若目标只看T，本pilot呈现trace接近的信号；按照事先声明的T和旧候选双参照晋级规则，它尚不能替换旧版。旧WL2也并非可靠标准。保持两参照有助于发现退步；正式论文的主参照、成功率非劣容许量、轨迹口径和统计方案应在新的独立验收前固定，不能根据现有结果追溯改门。

相邻两轮已完成28＋16＝44条合格完整episode，但分属不同条件与不同干预，不能混成44个独立配对或一个方案的44次成功率。最新质量轮可与他人共享GPU，墙钟只计资源，不提供新速度结论。15/16曾在历史target-only出现，称开发初态留出，不能声称全局未见测试集。

已有训练探索也给出约束：采集5796条同查询教师标签（3906训练、1890整episode验证），DATA/STEP/NOGAP和保持旧接受身份等候选没有稳定提高完整任务质量。提高接受率、换更大的draft或训练集拟合更好，均不足以保证真实轨迹；状态执行后会改变未来观察。WAM本身通常也需闭环评估，不能用离线训练代替任务成功与步数验证。

**现在试图解决的三个问题。**

1. 局部方向/状态误差如何转成可用的任务风险控制。WL2/MSE/RES有原文形式依据，具体阈值和5%/10%开发门属于本项目设置，没有直接继承成功率定理。对确定性BF16实现，零阈值批验证也曾与原顺序target产生不同动作；严格trace相同需要完整后继、数值形状和状态递推一致。
2. verifier成本能否随候选共享下降。7逻辑节点/14CFG行的“一次前向”不等于一次target状态的算力。共享KV确实有效；新增prompt/cross-KV缓存相对S0曾增加约6.48%–23.79%耗时，未接入。MegaTransformer要从实际verify瓶颈选一个原型，并证明完整policy获益。
3. 推理收益如何传到完整任务。多走一个chunk会抵消局部节省；WAM reset RPC在近期12任务中平均约125秒，约占客户端总耗时41.6%，包含prompt编码与缓存准备，子项尚未分解。它不是LIBERO物理reset。非模型RPC剩余约6.33%只是含物理/客户端的上界，不能推断Flash会显著加速整个WAM任务。

文献与边界：[接受判据原文](ACCEPTANCE-CRITERIA-CCFA-20261014.md)、[训练/严格后继与周期机制](TRACE-SUCCESS-CONTROL-RESULT-20261014.md)、[prompt共享负结果](PROMPT-KV-GPU-RESULT-20261013.md)。现有工作已覆盖连续状态投机、周期完整路径和缓存复用；“把投机用于WAM”单独不足以证明新颖性。可发展的贡献是有闭环证据的接受/回退控制，以及面向实际候选依赖与共享前缀的verifier优化。目前两者尚未组成完整方法与广泛实验。

**后续按优先级收口。** 现在继续讨论、论文结构、既有失败案例和协议整理，资源保持暂停。恢复实验前先冻结一个主候选、主参照及独立验收划分，优先跨更多Long任务与新初态检验成功/步数/长尾；原T和上一版均保留，不在同一组数据上反复选择阈值再称测试。训练若继续，优先补候选实际访问状态的teacher监督和完整chunk终点/执行动作约束；阶段性回退需能使用部署可得信号，simulator特权状态可用于诊断。上述方法是下一步待检验方案，不是本轮已经成功的方案。

速度线首先保留已对齐的共享KV与O1/O2/O3，单需求测真实verify及reset成本，再限定一个MegaTransformer原型；算法质量与系统精度分别验收，最后做组合的完整chunk及完整任务计时。暂不扩大到video投机、真实大模型联训、RL或全Transformer重写。Flash仅作为独立后端候选：当前_data接口不兼容，真实GPU动力学/SR等价与提速均未运行，迁移不作为当前论文前置依赖。

GPU累计预算已用8.355小时/10小时，余1.645小时；该额度不能视为足够完成正式benchmark和新kernel工作。此次暂停覆盖之前继续/共享GPU授权，未经后续恢复指令不启动任何实验。新排期需要匹配可用计算窗口和所需预算。

**时间判断：有条件赶得上，风险高。** 官方注册11/10 AoE、全文11/16 AoE；[官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)于10/14再次核对。老师完整稿维持11/01 18:00北京时间，距10/14约18天，内部全文提交11/15。正文官网当前仅明确日期，精确系统时刻仍需正式提交前核对。

| 节点（北京时间） | 可审阅交付与判断 |
|---|---|
| 10/17 | 冻结一个有限研究候选及质量协议；问题/方法/相关工作已有文字。若候选仍不稳定，及时缩题，不继续盲扫阈值 |
| 10/21 | 更广任务质量主表与核心baseline；据结果判断是否有足够算法贡献。资源暂停期间日历不会自动顺延 |
| 10/26 | 一个MegaTransformer verifier原型完成数值与完整policy/E2E验收；没完成则退出这次主文承诺 |
| 10/30 | 主证据、消融与图表冻结 |
| 11/01 18:00 | 给老师完整可阅读PDF，而非只有结果表 |
| 11/15 | 内部提交目标 |

要同时完成广泛质量证明、完整MegaTransformer系统与Flash迁移，现有排期和预算缺乏支撑。若10/21仍没有稳定质量或清楚的新贡献，应与老师讨论缩小论文范围或调整投稿节点；不能用工程pilot填补主实验缺口。提交日期赶上不等于达到接收标准。[详细倒排](CVPR2027-SCHEDULE.md)。

**本轮代码与HW整理。** 新增[代码入口图](../../../投机采样/docs/CODE-MAP-20261014.md)、[清理/归档提案](CLEANUP-PROPOSAL-20261014.md)和本页导航；HW只追加本研究记录和导航，历史内容保留。没有移动、删除、压缩代码或数据。原HW M01–M04完成于9/19，只表示既有里程碑；本研究及论文仍未完成。较早报告中的“下一步”只表示当时计划，以本页和最新原始报告为准。

文件盘点仅元数据；依赖核查仅文本，没有torch/CUDA加载或大张量重算。静态引用闭包涉及36个实验目录（含运行、预检、历史核查），因此v1/v2不能按名字清空。所有旧代码、失败证据、checkpoint、当前输入及tile方向保留；用户批准具体清单后才考虑删除或压缩。
