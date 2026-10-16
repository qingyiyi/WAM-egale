# 投机采样研究主线与训练思路

> **2026-10-15 17:41 追加：** 本机2/3完成，TF64/MLP96同预算六组预训练完成；8条完整任务证明最终T5 bank在两task/T与A_off上动作、环境、KV/VAE、RNG及2254组全部保存输出值一致。共享GPU的bank reset内部24.8–27.5ms、RPC103–128ms，不是新正式速度。Long10四臂40完整DEV运行中，结构未定；14997辅助FIT来源已纠正为A_off+历史研究TF混合。新结果/后处理失败恢复/AutoDL后续与清理见[本机2→3→4报告](LOCAL-234-RESULT-20261015.md)，时间表见[CVPR倒排](LOCAL-234-AND-CVPR-SCHEDULE-20261015.md)。下方当时“bank未在线、tensor待审、训练未完成”等均为历史状态。

> **2026-10-15 15:56 追加：** [总报告第9节](PROJECT-STATUS-E2E-AND-THROUGHPUT-20261015.md)明确已有MLP/TF离线结构对照、未完成的闭环选型，以及PF2与K2/B2/bonus的区别。Long10原T5 CPU bank11/11约44MiB及缓存CPU检查完成；T5占文本编码99.97%，裁padding0/11逐位一致不采用，在线reset/E2E仍未验收。AutoDL22/100仅T/A_off质量，0新GPU，原tile不改。

> **2026-10-15 15:10 北京时间完整入口：** [项目完整汇报：E2E、所有优化、实验吞吐与计划](PROJECT-STATUS-E2E-AND-THROUGHPUT-20261015.md)。任务运行1.151×与冷启动1.091×已明确端点；reset占原任务46.81%，policy占44.20%。AutoDL15:09到14/100、无新TF训练；下一版优先常驻服务、T5最终embedding缓存与离线teacher复用，均未宣称实测收益。CPU复算和图表已保存；原tile不改。以下旧“最新”均按日期视为历史。

> **2026-10-15 14:46 北京时间最新入口：** [本轮本机/AutoDL/清理/下一步总报告](LOCAL-AUTODL-CLEANUP-AND-NEXT-PLAN-20261015.md)。冻结 A_off 的多 task 正式 E2E 已完成 task0/1/5/9 各一对，双方 4/4 成功，自然 episode 合计 1.150692×，步数 1054/1044；独占审计和进程清理通过。先前单 task 双方成功三对汇总 1.188861×，T 失败的第4对不作为速度收益。AutoDL 的 Long10 50 对正式质量测试 14:37 到 9/100 条，预计 10/16 凌晨完成；旧 194 采集在 10 条后失败，不能称正在全量训练。旧 target-only 492 条完整记录逐一绑定服务端目录，TRAIN194/3672chunks 是离线教师候选，仍缺 same-query 数值证明。项目盘约 149 GiB，tile 方向未动。以下按日期保留历史状态。

> 最新早间结果（2026-10-15T09:54:10+08:00）：Long10用途登记、task2–9/init17新增9120条TRAIN标签与独审、TF64/MLP96各三seed CPU训练完成；六组均未过旧六门，保留A_off。投机质量仍需独立DEV/TEST；算子可基于冻结A_off隔离研究。见[早间统一报告](MORNING-LONG10-123-RESULT-20261015.md)，HW记录 CHAT-20261015-MORNING-LONG10-123-RESULT。


> 最新夜间结果（2026-10-15T05:45:28+08:00）：45次训练无稳定新权重，保留A_off；30条开发质量任务、新TRAIN与PF2/3/4已独审。算子融合已撤回，原tile不改。见[统一报告](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[扩大准备](NEXT-SCALE-READINESS-20261015.md)，HW记录 CHAT-20261015-OVERNIGHT-DRAFT-ACCEPTANCE-RESULT。


> **2026-10-14 研究追加（CHAT-20261014-DRAFT-DATA-VERIFIER-FOLLOWUP）**：历史Long4802行重建、15组CPU训练、6份真实profile、64个CUDA数值案例及6个完整chunk验收已完成；新draft未通过多seed原准入，保留A_off。四处融合数值一致，正式性能待独占；42chunk私有导入入口已封存。累计GPU31065.8421秒/147346 intents，余82.24分钟，自有训练/GPU已退出。详细结果与下一步见 [本轮报告](DRAFT-DATA-VERIFIER-RESULT-20261014.md)。以下原报告与历史状态保留。


> 最新结果（2026-10-14T20:10:22.540+08:00）：[18组draft结构初筛与verifier状态](DRAFT-ARCHITECTURE-RESULT-20261014.md)：Transformer在6/6组公平配对中优于容量相近MLP，但全部候选未过准入，保留A_off；低学习率root中位515/630对527/630，退化集中中sigma root。先核验旧train来源并扩大有效数据。verifier profile代码/CPU预检完成，GPU外来占用未启动，尚无融合原型或新性能收益。本轮0新GPU、余90.83分钟，自有训练/监测已结束；tile及preset保留。Knowledge：CHAT-20261014-DRAFT-ARCHITECTURE-SCREEN-RESULT。下方“尚未训练”等为历史时点。

> 最新安排（2026-10-14T19:16:40+08:00）：[draft选型与verifier并行计划](DRAFT-STRUCTURE-PARALLEL-PLAN-20261014.md)：先冻结接口，有限对照三类小候选，筛选首版结构后扩大训练分区；系统线可同时profile当前CFG14并做隔离block原型。现有训练便宜，优先补数据覆盖与闭环证据；旧Transformer shadow不能选定当前结构。已完成静态复核，0新GPU，尚未启动新训练/profile；余GPU90.83分钟。tile和当前preset不改。Knowledge：CHAT-20261014-DRAFT-STRUCTURE-PARALLEL-PLAN。

> 最新完成（2026-10-14T18:53:35.594+08:00）：[三档完整policy chunk实测](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)：短/中/长KV加速1.602×/1.544×/1.500×，45完整chunk、18/18正式配对胜出，同算法现场数值/KV/RNG及独立CPU审计通过。两次失败保留，最终累计GPU30550.488秒/36000秒（余90.83分钟），前向收费144550；自有进程清理完成。此前“待办/本次未测得”仅为当时状态，以本结果为准。[MegaTransformer整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已固定范围，下一步profile和单block原型；tile代码/HW不改，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-LENGTHS-COMPLETED。

> 最新更新（2026-10-14T18:32:43.139+08:00）：[当前完整chunk测试状态](CURRENT-CHUNK-LENGTHS-STATUS-20261014.md)与[MegaTransformer/tile整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已归档。三档入口/CPU预检完成；本机实测模型加载时外来lhw GPU进程出现，中止并清理，0完整chunk/0性能样本。累计GPU30110.927秒/36000秒，前向保守收费136964；下一新job继承本次失败账本。统一研究目标，tile原源码/HW保留，MegaTransformer仅固定首版范围，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-ATTEMPT-TILE-INTEGRATION。

> 最新安排（2026-10-14T17:40:03.895+08:00）：[总进度、待办及并行依赖](PROGRESS-AND-PARALLEL-PLAN-20261014.md)。短/中/长KV当前完整chunk实测已按用户最新指令改为待办；本机GPU正被他人使用，仅查询状态，0模型/0测试/0新增代码，不自动排队。下一测试资源已选本机独占窗口，无需重问偏好。算法质量、verifier分析设计与会话写作可并行，正式GPU计时须单独窗口；reset后置。当前CPU一核轻量整理，tile/旧质量门/preset不变。Knowledge：CHAT-20261014-SPECULATIVE-PROGRESS-PARALLEL-BENCHMARK-DEFERRED。

> 投机主线优先级更新（2026-10-14T17:28:57.196+08:00）：先完成投机质量与短/中/长KV完整chunk证据，再推进基于profile的MegaTransformer verifier；reset优化后置，有余时再做，仍计入完整任务成本。已测无reset长KV完整chunk相对T约1.48×，仅两个状态，非新PF2/总体收益。[决策与复算](RESET-DEFERRED-SPECULATIVE-PRIORITY-20261014.md)。当前本地GPU暂停、CPU一核；preset/旧质量门/tile不改。Knowledge：CHAT-20261014-SPECULATIVE-RESET-DEFERRED-PRIORITY。

> 投机调研更新（2026-10-14T17:09:15.336+08:00）：[轨迹评测与AutoDL方案](TRAJECTORY-AUTODL-PAPER-20261014.md)已整理；历史policy/reset约49%/42%，当前共享KV硬限A100 SM80/108SM，优先整卡A100尽快验证，H800需适配。近似算法无需逐条等长，仍测成功率/步数/总耗时，旧质量门不回改。用户要求中文稿会话逐段讨论、自行整理飞书；本轮两份初稿已按明确指令删除，不再生成本地论文稿。当前无新GPU/远端任务，本地CPU一核；旧清理仍待确认，tile内容不改。Knowledge：CHAT-20261014-SPECULATIVE-TRAJECTORY-AUTODL-WRITING。

> 投机采样当前安排（2026-10-14T16:42:00+08:00）：用户要求暂不使用GPU、CPU最多一个核；当前无实验/排队。整体判断、已测收益、CVPR风险见[当前状态](CURRENT-STATUS-20261014.md)；[代码导航](../../../投机采样/docs/CODE-MAP-20261014.md)；[待确认清理清单](CLEANUP-PROPOSAL-20261014.md)。本轮只做低资源盘点与文档整理，未移动/删除/压缩；tile方向内容保留。Knowledge：CHAT-20261014-SPECULATIVE-OVERVIEW-CLEANUP-PROPOSAL。


更新日期：2026-10-14（Asia/Shanghai）  
研究范围：第一代公开 LingBot-VA 的 action 分支；draft → 真实 target/verifier → accept/rollback。  
当前阶段：固定PF2的WL2/MSE/真实scheduler残差比较完成：20完整GPU数值chunk与16完整任务独立复核；T:3/4,steps=1640; PF2_WL2:4/4,steps=1144; PF2_MSE:3/4,steps=1655; PF2_RES:4/4,steps=1207；结果仅为开发pilot；Flash-LIBERO代码审查发现当前客户端_data不兼容，尚无物理/SR等价实测；CVPR2027老师稿定11/01 18:00北京时间。 既有28任务结果保留，本轮另有16任务；详见第154节。

## 1. 记录归属与研究目标

- 本文件是 HW 中投机采样研究思路的持续整理入口；完整证据保留在原报告和 Knowledge Ledger 中。
- 后续讨论按“已观测事实、待验证假设、候选方法、实验结果、决策变化”更新本文件，并新增 Knowledge record、增量同步 indexes/compact（保留旧条目）、追加 PROGRESS。禁止把讨论或建议直接写成已完成实验。
- 用户同意逐步推进：每一步输出本步完成程度、结果、未解决项和下一步要做什么；尚未完成的验证不能跳过或写成已经证明。
- 使用 `speculative-sampling`、`action-only`、`libero` 等明确标签。不把本路线并入 tile/macro-tile、CUDA 多流、Attention/MoE/Q-Bridge 的算子研究；旧架构和旧算子里程碑不代表本路线的进度。
- 工程组件可以引用，但 tile 路线的收益、失败或完成状态不能充当投机采样的实证结论。Video draft 虽有离线训练，video 投机闭环仍延期。
- 目标不是最小化平均 L2 或最大化接受率，而是：在任务成功率不劣、环境步数相近且长尾不过度增加的条件下，降低完整 episode 的 wall-clock。
- “相同准确率”指任务成功率的非劣性，不指每个 latent/方向逐元素相同。容许差异 ε、步数容许增量 δ、长尾与失败处理方式须在正式比较前约定；目前没有指定其数值。
- 初次整理阶段仅记录研究假设；此后用户在第13节明确授权最多20完整episode及必要默认关闭诊断/小模型训练代码，现已完成。授权不扩展到无限训练/RL/真实联训；本次第14节仅分析既有数据，无新源码/GPU/rollout。

## 2. 已观测事实及其边界

| 事实 | 已有证据 | 不能推出的结论 |
|---|---|---|
| 历史单 chunk 确实有加速 | 稳态 profile：action total 4.4360→2.8211s；policy total wall 6.2772→4.6559s，约降 25.83% | 每种 init、每轮线上配置都同样更快；这些 profile 数字也不能与另一 campaign 的 E2E 直接混算 |
| 较早的真实四-init 配对中，单次 policy 更便宜但轨迹更长 | aggregate policy/chunks 约 11.384→10.182s；env steps 276.5→314.25；E2E 只快 0.34% | 多走环境步一定是 L2、接受深度或 fallback 引起的 |
| 最新六-init 描述性池仍未加速 | 两臂均 6/6 success；K2 E2E +0.89%、env steps +9.37%、chunks +10.09% | 已证明成功率非劣或跨任务有效；单任务小样本不足以支持这些主张 |
| 最新新增 init4–5 的单次 policy 也有波动 | aggregate policy/chunks 约 11.215→11.828s，K2 约慢 5.46%；E2E +3.37%、env steps +1.05% | 当前只需解决环境步数，verifier 工作预算无需再检查 |
| L2 能解释现有 verifier 的接受/拒绝 | 它是同一 candidate state/transition 的局部边代价，部分量还直接参与接受规则 | L2 低就能预测任务成功、任务进度或剩余环境步数 |
| accepted depth/fallback 暂无可靠的长尾解释力 | 历史审计只有 10 个独立 episode；accepted depth 对 env steps 的 Pearson -0.1095、Spearman 0.0673 | 能用 171 个 chunk 或 4,600 个 decision 伪装成独立样本来训练可靠 gate |
| 新训练候选与最新线上配置不同 | Phase A Transformer 的 shadow 两步接受代理指标为 90.67%；最新 Phase 2B 使用 9 月 29 日的 beam-2 checkpoint | 最佳 shadow 候选已经完成真实环境训练，或已经在线验证失败/成功 |
| 大小模型同步训练只做过代理实验 | surrogate target 拟合固定标签、draft 跟随其输出；没有真实 WAM target 更新 | 大小模型真实联训已经证明更短轨迹或更快 E2E |

上述单次 policy 推算采用各 campaign 的累计 policy 时间除以累计 chunk 数，不是同状态、同阶段的严格 matched 单 chunk 因果比较。历史 profile、干净 timing 和诊断数据分别报告。

依据：[单 chunk 与 E2E 成本账本](../../debug/Debug-003-K2-E2E-bottleneck-20261008.md)、[L2 与任务指标](../../debug/Debug-004-L2-acceptance-env-steps-20261008.md)、[历史信号审计](../../debug/Debug-008-Phase2a-signal-audit-20261009.md)、[最新闭环对照](../../debug/Debug-011-Phase2b-matched-closed-loop-20261009.md)、[同步训练代理](../../debug/Debug-005-draft-synchronous-training-proxy-20261008.md)。

## 3. 当前核心问题与待验证假设

环境步数取决于执行动作如何改变物理状态，以及何时满足任务完成条件。去噪 scheduler updates 与机器人 env steps 不是同一单位。当前近似验收不提供整条动作轨迹与 target-only 数值一致的保证。

| 编号 | 待验证假设 | 应观察的信号与对照 |
|---|---|---|
| H1 动作语义比全局 L2 更重要 | 抓取时机、夹爪开合、关键位姿或接触误差可能比平均 latent 误差更影响完成步数 | 已执行动作、夹爪切换、接触/抓取/放置里程碑、task-specific 成功谓词；不能仅比较 direction/latent L2 |
| H2 小偏差造成反复纠正 | 接受的微小误差可能在跨 chunk 重规划中累积成停滞、抖动或返工 | 动作方向反复、位姿/物体状态变化、阶段回退、同阶段停留时间；按同 task/init、相同初始条件比较 |
| H3 训练分布与线上状态不一致 | 离线 target-only 状态与 draft+verifier 实际访问的环境状态、候选去噪状态可能有偏移 | 在线访问状态上的冻结 target 标签，尤其是返工/困难阶段；必须使用同一实际 candidate state，而非另一条 target 轨迹的标签 |
| H4 误差在阶段和维度上不均匀重要 | 早期/晚期去噪、接触前后、不同动作维度对控制结果的敏感度可能不同 | 分阶段、分维度的执行动作差异与后续真实进度；early shadow 接受率低只是线索，不是任务敏感性的因果结论 |
| H5 实现或评测合同导致差异 | cache、sigma/timestep、动作前缀、执行 stride、成功检查时机等错配也可能改变步数 | 先审查合同与路径一致性，不能先把轨迹变长全部归因于模型训练 |
| H6 verifier 的有效工作覆盖仍不足 | leaf 输出没有被正确消费或批量成本过高，可能抵消局部收益 | 分开审查普通 K2 与 K2+bonus：7-row 一次验证并消费选中 leaf，对比 3-row 验证后另跑 target；比较实际 target work 与 wall-clock |

连续 goal distance、task progress 或价值函数尚未被现有数据证实可用。优先使用实际可读且任务语义明确的 simulator state/predicate；缺失字段保持缺失，不用稀疏终点 reward 冒充连续进度。特权状态可作诊断或训练辅助，但不能悄悄成为只在 simulator 中可用的部署输入。

相关性用于筛选假设，不足以归因。若未来环境可被完整恢复，才设计同一 simulator/observation/cache/noise 状态下的 target 与 speculative 反事实分支；snapshot/恢复能力尚待验证，不假定已经具备。

## 4. 弱化 L2 的准确含义

1. L2 降为局部解释指标或训练辅助约束，不再作为唯一晋级目标，也不因 L2 变小就声称任务更快。
2. 现有 verifier 仍使用局部 cost 验收。“弱化 L2”不等于直接放宽阈值、删除验收或取消 target fallback。
3. 若替换 gate、改变 threshold 或引入 task-aware 权重，属于独立方法变更，需单独验证成功率、步数和长尾，不能与 draft 训练同时改变后再归因。
4. 分层看待指标：success/失败风险 → env steps 与长尾 → E2E；accepted prefix、target rows/forwards、draft/verifier 成本与 L2 用来解释结果。

## 5. 训练路线比较

“真实 LIBERO”指实际运行 LIBERO simulator 的任务闭环，而非离线 trace、shadow replay 或物理机器人实验。环境反馈训练和 target 监督不是非此即彼。

| 路线 | target 是否更新 | 小模型训练信号 | 当前状态与用途 |
|---|---|---|---|
| A 离线蒸馏 | 冻结 | 保存的 target 输出/多步状态 | 已有基线；便宜、可复现，但不能单靠平均 L2 保证任务行为 |
| B 真实闭环分布上的在线蒸馏 | 冻结 | 系统实际访问状态上的 target 标签，可复用 verifier 已算输出 | 已完成固定S0访问分布上的单轮B_on监督pilot，未见稳定任务优势；非最终candidate自采样迭代或RL |
| C 冻结 target 的任务反馈训练 | 冻结 | 组合系统的 success、env steps、实际推理成本；保留必要 teacher/验收约束 | 待设计和验证；可以是混合监督、轨迹偏好或受约束 RL，不预设 PPO 可直接套用 |
| D 真实大小模型协同训练 | 更新 | task 目标 + 小模型可预测性/验证成本等 | 后续探索，未开展；新 target 必须重建 target-only 基线，并验证原能力是否受损 |

### 优先方向的定义

建议先固定 target checkpoint、verifier/scheduler 协议和评测合同，只更新 draft 参数。在实际 LIBERO 中运行的仍是 **draft + 真实 target/verifier + accept/rollback** 整体，而不是让小模型脱离大模型单独学控制策略。

训练与模型选择应服务于：

```text
最小化完整 episode 的 E2E 时间
约束：任务成功率不低于 target-only（预先约定非劣容差）
      配对环境步数相近或更少，严重长尾不增加
      target/verifier 协议与比较条件明确且可追溯
```

不能只奖励“少走步”：提前失败也可能走很少步。失败必须按预注册规则计入成功率及成本；报告全部 episode，同时单列双方成功的配对步数，不能只筛成功案例宣称改善。

必要的 target 监督是参考/约束，不是任务效用的唯一目标。完全放弃 teacher 一致性也可能让 task-reward 看似较好的 draft 大量被 verifier 拒绝，既不改善实际动作，也不减少计算。拒绝分支对最终 reward 的信用分配、稀疏任务奖励和去噪策略的概率/梯度定义，都是采用 RL 前需解决的问题。

在线蒸馏的启发来自 DAgger 对策略自身访问分布的处理，以及 Online Speculative Decoding 对 draft 的在线适配；这些论文不直接证明本 WAM 路线有效。[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)、[Online Speculative Decoding 原论文](https://arxiv.org/abs/2310.07177)。

DPPO 表明扩散策略可以通过任务反馈做 RL 微调，但不能据此假定当前 flow/scheduler + 离散 accept/rollback 的实现能直接接 PPO，或必然减少 LIBERO 步数。[DPPO 原论文](https://arxiv.org/abs/2409.00588)。

### 协同训练需拆成两类

- **推理协同、训练只更新 draft**：大模型已经通过 verifier/fallback 与小模型协同；这是当前协议，也是应先验证的训练方向。
- **大小模型权重都更新**：改变了 teacher、verifier 和任务策略，会把“teacher 变强/变弱”与“draft 加速”混在一起。需保留原始 target、更新后的 target-only、更新后的组合系统三组，分离任务能力变化与投机收益。可研究但不是当前证据支持的第一优先级。

同步代理实验不是上述第二类真实联训，不能用其递归误差改善替代闭环证明。

## 6. 接下来要回答的问题和验证顺序

用户已同意按以下研究顺序逐步推进，并要求每步提醒后续工作。已完成第1步静态/CPU核查、第2a/2b/2c保存数据诊断，以及用户授权的最小冻结target采集/同预算训练/16开发集评测。当前追加夜间init3/init4行为与成本诊断；物理/模型根因和同负载target work仍未完成，新实验/代码/服务继续按hybrid与单独预算边界处理。

1. **合同与成本审查**：先排除 H5，并厘清 H6 的 leaf 复用。单独测量有效 target work；bonus 覆盖的是额外 scheduler update，不直接是 env step。
2. **任务进度诊断**：在少量匹配的完整 episode 中找出最先出现停滞、返工或关键阶段延迟的位置。诊断与干净 timing 分开，不用单 chunk/smoke 或截断结果代替任务证据。
3. **最小训练路线对照设计**：A 离线基线、B 冻结 target 的真实访问分布蒸馏、C 冻结 target 的环境反馈混合训练。保持 checkpoint/协议一致，按 task/init 划分 held-out 集，记录数据、更新步数和交互/训练预算；先验证最小假设，不直接展开三组长训练。
4. **任务级选择**：同时报告 success、配对 env steps、median/长尾与 E2E；L2/接受率不单独决定晋级。新 draft 的 shadow 排名不能替代真实组合系统验证。
5. **再决定是否研究 D**：仅当冻结 target 路线的限制有证据、预算与问题明确时，设计真实联训；否则优先收束现有路线或保留负结果。

新采集/评测须遵守 EGL + 独立并行 worker；单 A100 上算法条件串行，完整可用 horizon，无人为截断，provenance 可追溯。模型代码修改、服务操作、训练和采集仍需各自明确启动决定。

## 7. 决策状态和更新规则

- **用户明确要求**：以后把投机采样思路记录在 HW；与 tile 算子路线分开；以成功率、相近环境步数和速度为研究目标，弱化 L2 的任务代理地位。
- **用户同意的研究顺序**：先冻结 target，先诊断环境步数相关机制，再比较真实闭环分布蒸馏与任务反馈训练；优先训练组合系统中的 draft；每步反馈结果并提醒下一步。
- **未决问题**：B/C 是否优于 A；什么进度信号可靠；是否有必要真实联训；leaf 复用的净收益；非劣容差与成本预算。
- **未开展工作**：真实LIBERO reward/任务反馈训练、RL与真实WAM target/draft联训。用户授权的20episode GPU最小监督对照及默认关闭诊断源码已完成；当前未启动额外实验。
- 旧报告仍保留其当时结论。此前“暂不继续长训练”不因讨论自动解除；第13节明确授权有限20episode最小对照，现已结束且未达联合目标。当前不扩大训练；未来方向变更需追加证据和明确决定，不回写历史为成功。

每次更新至少记录：问题、假设、证据及反证、方法范围、与基线相比的评测合同、完成/未完成状态、下一步、来源。Knowledge compact 仅承载入口与关键结论，不复制全篇。

## 8. 逐步推进状态

| 步骤 | 当前状态 | 产物或下一项工作 |
|---|---|---|
| 1 合同与 leaf 复用核查 | 静态/CPU 核查完成，32 tests passed；真实 WAM 数值与净成本未测 | 普通 K2 不消费 leaf，bonus 全接受才消费；近似验收不保证 target-only 轨迹等价 |
| 2 任务进度诊断 | 复现/开关关卡通过，600步第二罐抓取返工阶段已定位；同init算法/物理因果仍未完成 | 4局同init T/S0已完成与CPU复核；阶段增减逐init记录，S0全测复现通过；成本/新训练仍另关卡 |
| 3 冻结 target 的最小训练对照 | 授权20episode完整最小对照已完成；observer/teacher及同预算A_off/B_on实现并验证 | 每臂833唯一样本、1000×512更新；B_on一轮S0访问分布teacher监督，非RL或最终candidate自采样迭代 |
| 4 完整任务指标验证 | 16完整development eval无联合收益；6局GPU不扰动及现有账本19 CPU检查PASS | 32对action+25.93%：短KV快/长KV慢；固定负载成本因果/自然E2E待另确认，不晋级或自动训练 |
| 5 是否真实联训 | 后续决策 | 冻结 target 路线的限制被证实后再决定；已有 surrogate proxy 不算完成 |

第 1 步报告：[静态合同与 leaf 复用核查](../../debug/Debug-012-speculative-step1-contract-leaf-audit-20261010.md)。本步未发现已审查路径上的候选持久 KV 写入、scheduler 隐式步索引推进或两臂执行 stride 差异；真实模型与所有未观察控制问题仍不能据此排除。新测试和 CPU 合成检查不构成 LIBERO 轨迹根因或 E2E 收益实证。

## 9. 第 2 步盘点与 RL＋KL 假设（2026-10-10T22:50:59+08:00）

- 已保存六个配对 init 的动作与 RGB，可先做离线行为诊断；没有可直接读取的物体位姿/连续 task progress，最终成功 chunk 的执行后 RGB 也未保存。数据存在不等于对齐完成或返工原因已知。
- RL 有可能把 draft 优化到组合系统的任务成功、env steps 与实际 E2E 成本；冻结 target/verifier，KL 只作定义明确的辅助约束，不直接保证轨迹长度。该候选未实现、未训练、未验证。
- 当前 draft 输出确定性连续方向，无现成密度/log_prob；固定同协方差 Gaussian KL 退化为加权平方 L2，不能靠改指标名称解决问题。大模型真实联训仍是后续独立问题。
- **下一步仍属第 2 步**：验证 episode/chunk/观测文件关联，恢复并截取实际执行动作，找首次可见停滞/返工；之后才确定最小训练对照与随机策略/KL 参考。

详情与原论文：[第 2 步数据及 RL/KL 候选说明](STEP2-DATA-AND-RL-KL.md)。

## 10. 用户方法确认与第 2b 步完成（2026-10-10T23:18:28+08:00）

- 用户同意冻结 target/verifier、任务反馈训练 draft 与 KL 辅助的方向；算法/概率定义、权重/预算/非劣门槛仍待设计，训练未启动，真实联训后置。方向确认不构成效果证据。
- 12 episode/229 chunk 已唯一关联；按当前源解码/执行合同重建 3507 条控制命令，剔除 48 条条件槽位与 109 条未执行尾部；安全读取全部 217 obs，索引 3424 组动作后双视角 RGB。
- 原生 75 chunk event 对账通过，旧 154 chunk 边界明确推导；旧 source 整文件不同，历史逐位实际命令等价未证。12 terminal chunk 共 83 个动作后 RGB 缺失，仍无物理位姿/连续进度。
- 六对 env-step 增量为 +172、−2、−9、−10、+3、+3，总计 +157；init0 主导总体增加。不能删除 init0 后晋级，也不能把此分解归因到 L2/夹爪/返工。
- **下一步第 2c**：优先对 init0 两臂整理视觉阶段与控制命令时间线，找首次可见停滞/返工；保留其余五对，不启动训练。

报告：[第 2b 步动作/观测对齐](STEP2B-ACTION-OBS-ALIGNMENT-20261010.md)；数据：[evidence](../../../review/OMISSIONS.md)。

## 11. 第 2c 步：首次搬运掉罐与重搬（2026-10-10T23:50:11+08:00）

- 保存数据的视觉子步骤完成：init0 K2 约 101–103 env-step 蓝黄罐脱离手部并回到桌面，随后空手到篮子、268 步再次搬运该外观罐；target 对应已查窗口中仍携罐搬运。支持具体返工过程，但物理/模型根因及全部 +172 步的量化归因未证。
- 掉落位置是 rollout chunk 6 / server frame 24 / env steps 93–108；98–104 两臂都发正闭合命令。K2 第一次打开命令到 136 步才出现，target 为 139；不支持简单“提前松手”解释。命令不等于实际夹爪开度/接触/稳定抓持。
- 全 12 episode 命令检查：init0 target/K2 正命令段为 2/3，其余 11 个 episode 均 2；另外五对尚未完整视觉诊断，不宣称它们没有掉落。人工对象外观身份和视觉入篮不冒充 simulator id 或原生成功谓词。
- 静态确认本任务两项 In（alphabet_soup_1、tomato_sauce_1 入 basket_1_contain_region）及任务类逐项求值；旧 RGB 没有真值，运行采集未验证。原生目标/真实抓持是进度诊断候选，不发明连续 goal-distance。
- 44 obs、696 组双视角、1392 图像 hash 复核；原始 807 runtime 文件、旧证据、源码和 tile 记录不变。没有训练、服务、GPU 或新 rollout。
- **下一步**：先设计最小冻结 target 对照和真实持稳/物体进度采集合同，验证字段可行性后再考虑训练；任务反馈/KL 仍为待验证候选，真实大小模型权重联训后置。

报告：[第 2c 步 init0 视觉返工诊断](STEP2C-INIT0-VISUAL-REWORK-20261010.md)；图像/命令证据：[evidence](../../../review/OMISSIONS.md)。

## 12. 第 3a 步：诊断与最小对照设计（2026-10-11T00:09:07+08:00）

- 第 3a 设计及静态/CPU 可行性核查完成，15 项窄源码语义检查通过；observer 未实现、真实字段未采集、训练未启动。下一步先确认默认关闭 client-only observer 的窄源码范围和 CPU 合同，不跳到采集/训练。
- 补充第 2c 原生谓词语义：basket region 为 SiteObjectState，check_contact 恒真，本任务 In 主要为区域包含，不是实际接触/持稳信号。对象姿态 getter 是可变 sim view，快照必须 copy；body wxyz 与 eef/site xyzw 不可混算。
- 诊断合同覆盖实际七维命令、夹爪 qpos、同 grip-site 位姿、两个物体位姿、双指接触、两项 native goal/done/reward 及 terminal；N 后状态加稳定化后初始状态。特权字段不进入模型，不能额外 step/forward/reset/cache 或改变默认行为。
- 线上 teacher 必须是实际 predictor 特征与同一 query state 的 target CFG 标签；compact 跳过 draft commit、非 full tree 为摘要，不能直接当完整线上数据，更不能 accepted draft 自己当 teacher。
- 推荐候选先真实 provider/不扰动，再 task0 init0/1 × target/current K2 四完整诊断；这只是设计，需另定预算/启动确认，不解除旧执行门限。真实访问分布训练需同预算 A_off/B_on 区分额外优化；C1 任务事件加权监督不是 RL，C2 RL/概率定义和真实联训继续待定。
- success/配对步数/长尾/E2E 仍为主判据，容差/预算/held-out 未定；init0–5 是已看 development，不冒充 pristine held-out。原始 runtime/旧证据/源码/state/rules 与 tile 记录保留。

报告：[第 3a 诊断合同与最小对照](STEP3A-DIAGNOSTIC-AND-MINIMAL-COMPARISON-20261011.md)；候选合同：[evidence](../../../review/OMISSIONS.md)。

## 13. 用户纠偏与最小冻结 target 夜间启动（2026-10-11T00:49:03+08:00）

- 用户明确授权必要源码/小模型训练与最多20完整episode自动关卡，纠正100episode纯基线偏离主线；旧任务尚未启动即撤下。该授权只适用本最小对照，不解锁无限长训练/RL/真实联训。
- 默认关闭observer/同query teacher与draft继续训练接口已实现，37 CPU合同/回归+9真实WAM模块CPU tests通过。真实LIBERO各16规定动作on/off轨迹/读取sim和Python/NumPy RNG一致，非完整WAM数值验证。
- 固定T/S0 init0/1共4采集；仅在全native行/terminal、teacher每树3实际节点与每chunk完整marker/query hash通过后，A_off/B_on同原B2/normalization/架构/loss/optimizer，1000更新×512、lr1e-4、共同唯一样本预算，固定末次checkpoint。B_on仍是S0真实访问状态的单轮teacher监督，非RL。
- T/S0/A_off/B_on×init2–5共16干净eval；本轮更新隔离但历史已看/训练暴露未排除，仅development，不宣称非劣/泛化。失败全部报告，L2不能单独晋级。
- 已后台运行running_collect_T，supervisor PID 1247023，独立tmux/PPID1。本机熄屏或SSH断开不作为运行依赖；服务器重启/OOM/磁盘失败不保证。前两次PTY全部重定向导致SIGHUP、无GPU启动，保留tmux stdin后修复。
- 预测仅为假设：访问分布拟合可能改善，任务步数/E2E未知。按实际结果自动报告后续方案；失败停止，不扩预算、不直接RL/联训。

启动与预登记：[报告](OVERNIGHT-MINIMAL-FROZEN-20261011.md)；状态：[status](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/status.json)；动态结果与下一步：[HW结果入口](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)。以上更新取代此前“observer未实现/本轮无新实验”的当前状态，旧第3a等历史原文保留。

## 14. 最小对照完成与失败/成本诊断（2026-10-11T07:50:14+08:00）

- 夜间20/20完整episode于02:04:39完成，约78.7分钟；GPU已释放。本次仅CPU分析，未启动新训练/rollout或改模型/gate/bonus。T4/4、S0 3/4、A_off4/4、B_on3/4，所有失败保留。init2–5历史已看/训练暴露未排除，仅development。
- A_off四对少1/13/9/4步，总1093对1120（−2.41%），但chunk全部仍17/19/19/17、总72。平均E2E377.66对344.36秒（+9.67%）；每局差值reset−3.47、policy+35.62、cache+1.91、残差−0.75秒。请求成本非纯GPU；少走步不自动省chunk。
- B_on init3约235–238步搬运第二罐可见掉落，期间仍正闭合命令；长尾反复尝试、796步失败。S0 init4对象未成功提离，手离开而罐留桌面，反复尝试至796失败。图像定位并非接触/物体pose/各In谓词或物理/模型根因证明。
- 16eval/349native chunk/5419执行命令/5296组双RGB对账。14成功末chunk缺123步RGB，两失败末chunk仍cache有terminal RGB；实际796步/sim801与公共795惩罚分开。
- 独立worker不是恒定双policy并发：S0 init5全部17chunk在peer完成后启动，policy/chunk5.784秒不能直接当相同负载算法收益；B_on init3后29chunk也在peer完成后运行。包络是需求代理，不是GPU利用率/精确请求重叠。
- 不继续盲目teacher训练、不晋级、不上RL/KL或真实联训；B_on仅固定S0真实访问分布一次teacher监督，不能称任务reward训练或全面否定在线蒸馏。
- **下一步建议另授权6完整episode原生诊断：T/S0/B_on×init3/4，每臂2独立EGL worker、算法条件串行，固定checkpoint/seed/threshold/horizon，无新训练/额外target forward。**先确认contact、pose、夹爪qpos、各In与terminal状态及旁路合同；完整WAM on/off数值等价仍未证明。之后单列同phase/并发需求T/A_off成本与target work计数设计，再决定敏感阶段/困难状态反馈训练。当前没有启动这6episode。

本步报告：[失败时间线与成本诊断](../../debug/Debug-013-speculative-minimal-frozen-failure-cost-20261011.md)；[证据目录](../../../review/OMISSIONS.md)；夜间结果：[HW结果入口](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)。旧第13节为当时启动记录，保留原文；当前状态以本节为准。

## 15. 用户授权6episode原生诊断与后台启动（2026-10-11T08:02:48+08:00）

- 用户明确同意6完整episode并询问时间；实际08:00:28启动，当前running_diag_T、supervisor PID 1796841。固定T/S0/B_on×init3/4，每臂2独立EGL worker/各1局、算法条件串行，完整horizon，无新训练/teacher/bonus/gate修改或成本复测。
- fresh16 CPU回归通过；运行包装6项scripted合同通过，明确native ON/teacher OFF；真实provider scripted on/off各16规定动作状态/读取RNG不扰动PASS，非完整WAM数值证明。复用现有源码，仅新增job包装。
- 预计运行35–50分钟，检查及初步整理45–70分钟（08:00启动估计08:45–09:10），只是估计；长尾或检查失败可能改变，不人为截断或自动加局。
- **新复现限制**：启动seed固定，但server reset不重设RNG，旧init3在init2后（T/S0/A_off17chunk、B_on16）；当前init3首局，未恢复旧CUDA RNG。旧init5前序也受init4长度影响。不能当严格同噪声反事实或把未复现失败归因于observer；新3臂按统一首局/worker合同。旧原始记录保留。
- 后台独立tmux spec-native6-075638/native6（daemon PPID1），保留PTY stdin，本机熄屏/SSH非依赖；服务器硬件/重启故障不保证。source/checkpoint/target-stat保护、磁盘/GPU门，仅自有子进程清理，无自动重试。
- 完成后自动核对native连续行/goal合取/native success/terminal RGB hash，输出逐步pose/contact/qpos/In和描述统计；contact不是已验证持稳标签，诊断不是clean timing，不晋级。
- **下一步**先CPU解释原生事件/是否复现旧失败，再另设计同负载T/A_off成本计数；不自动RL/KL/真实联训或追加实验。物理/模型因果仍未完成。

[启动报告](NATIVE-DIAGNOSTIC-6EP-20261011.md)；[实时状态](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/status.json)；[HW自动结果](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/FINAL_REPORT.md)。

## 16. 六局原生结果检查与复现关卡（2026-10-11T11:55:54+08:00）

第15节保留当时启动/估计，不代表当前仍运行。6局08:23:31结束、23.05分钟、全部成功；GPU显存0MiB。本次用户要求先检查，只CPU分析，没有启动任何新实验、训练或源码修改。

- T init3/4=293/299步、19/19chunks；S0=298/310、19/20；B_on=280/309、18/20。合计T592/38、S0608/39(+2.70%步)、B_on589/38(−0.51%步)，只两init开发性描述，不证明非劣或B_on晋级。诊断耗时不是clean timing。
- 115chunk/1789实际命令/1795native行/1720双视角观测及6终止NPZ对账通过，解码命令逐值等于native，goal合取及EGL/teacher关闭通过。69个terminal-chunk动作无server RGB，不用6终止快照冒充补齐。独立CPU复核666去重文件hash不变。
- **旧S0 init4、B_on init3两条796步失败未复现**。init3旧第二局/新首局，step1命令/图像已不同；init4同首局，前三臂76命令及前5完整action/video张量逐位一致，但T61/B_on62/S073步RGB先不同，step77才命令分叉。S0 step73 wrist仅3像素值差1；没有因果归因observer或3像素。
- 当前新三臂初始已采字段差最大6.20e−17，接近浮点舍入，不作显著初态差异论据。完整模拟器快照、隐藏状态、WAM ON/OFF数值等价均未证明。
- 双对象各一段双指接触并抬升，非已验证持稳分类器；S0 init3 soup In154–161真、162–166假、167起真，首次时仍闭合夹持。T/S0 init3终止sauce仍双指接触，原生成功为区域包含，不证明释放落稳。不把首次In/二值接触直接作为RL奖励。
- **下一步顺序调整**：先CPU审查复现与provider读取合同；另授权固定起始RNG/可支持模拟器状态/peer负载的OFF/OFF重复基线及observer ON/OFF最小policy关卡（完整恢复工具尚未实现，必要改动另确认）；再单独同phase/同需求T/A_off成本计数，最后决定冻结target反馈训练。无自动新GPU/加局/RL/KL或联训。

[本次检查报告](../../debug/Debug-014-speculative-native-six-results-reproduction-gate-20261011.md)；[独立复核](../../../review/OMISSIONS.md)；[证据目录](../../../review/OMISSIONS.md)；[自动结果原件](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/FINAL_REPORT.md)。

## 17. 复现第一关通过与6局完整策略已获确认启动（2026-10-11T12:13:04+08:00）

- 当前探索：冻结target下的组合系统，成功率不劣/环境步数相近时整局更快；单chunk历史加速已证但整局收益不稳定，最小A_off/B_on对照未达联合目标，旧两次失败未复现。不是tile研究，也不是L2最小化。
- 本步CPU源码审查查明：正式client本文件未显式seed Python/NumPy/env，旧短脚本有三种seed；server seed不应用另一进程。flattened init只有time/qpos/qvel，不是完整hidden snapshot。缺口存在但旧失败原因未证，公共源未改。
- 已做两独立EGL worker固定保存动作三条件回放（S0 init3/4），12:02–12:03约1.82分钟，共1824动作，OFF/OFF与OFF/ON状态/RNG/RGB hash/终止一致；610即时capture检查、20字段36600数组hash复核通过。0自适应policy episode/0模型forward/0更新，不当任务准确率或WAM完整数值证明。
- 用户明确“确认，按6局上限执行”，允许job-only种子/指纹包装；13既有CPU回归/16scripted包装检查通过，2026-10-11T12:10:35.560799+08:00在独立tmux实际启动完整策略同S0+冻结target、OFF-A/OFF-B/ON×init3/4、每条件2独立EGL worker，当前running_repeat_off_A。首局和client seed固定，所有条件相同指纹记录，只有ON native；无teacher/训练/RL/联训/bonus/gate更改。
- OFF/OFF第一分叉若存在则四局后停止、不运行ON，不自动重试/加局；完整原horizon无人工截断。新source manifest34项/target9文件size-mtime、GPU/磁盘/互斥锁、端口/EGL审计，后台PTY保留不依赖本机屏幕/SSH。运行预计25–45分钟非保证。
- **下一步**完成正式重复/开关关卡并CPU核对观测/动作/video保存张量，定位最早分叉；通过后另行同负载T/A_off成本，再决定冻结target反馈训练。完整server RNG/cache恢复、旧失败根因、非劣和任务训练收益仍未证。

[本步报告](../../debug/Debug-015-speculative-repro-seed-replay-gate-20261011.md)；[回放证据](../../../review/OMISSIONS.md)；[策略运行状态](../../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/status.json)；[范围计划](../../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/plan.json)；[自动结果](../../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/FINAL_REPORT.md)。

## 18. 六局完整策略复现结果：先OFF/OFF，再ON/OFF（2026-10-11T12:46:52+08:00）

第17节保留启动时状态；本任务实际12:41:37结束、31.03分钟、6/6成功，没有追加局数。用户要求定时检查，本轮实际6次状态轮询跟踪至结束；没有可用常驻automation接口，未创建或虚构离线定时推送。

| 条件 | init3步/chunk | init4步/chunk | success |
|---|---:|---:|---:|
| OFF-A | 293/19 | 600/38 | 2/2 |
| OFF-B | 293/19 | 600/38 | 2/2 |
| ON | 293/19 | 600/38 | 2/2 |

- **先OFF/OFF**：两个init初始20类NPZ逐位相同，全部已测状态/RNG/双RGB/实际命令/done/reward和终止长度相同，两个OFF对应57chunk的完整30通道action与video latent逐位相同；自动关卡通过后才启动ON。
- **再ON/OFF**：ON与OFF-A两个init全程同样无已测分叉；另外57chunk的全部保存action/video相同。读取前后state/RNG/RGB检查无assert，特权信息未进入policy输入。只证明本S0/target/两init/共同插桩合同下未观察到扰动，不是全任务、全seed或完整server/sim隐藏状态数值证明。
- CPU按实际BF16部署解码核对171chunk/2679执行命令，2685连续指纹行、2616组server双RGB、895 ON native行及6 terminal快照；4条件-init比较/114chunk对action-video相同。另一CPU脚本复核CSV/边界/命令/比较汇总与721去重文件hash，source34与target9文件stat合同不变。63终止chunk动作没有server RGB，不用6 terminal快照冒充补齐。
- 本轮同一S0，不含T/候选训练比较；6/6成功不代表训练改善、正式非劣、旧失败修复或整局加速。共同指纹/ON native含诊断开销，旧launcher latency_benchmark标记不使其成为clean timing。旧client seed缺口虽真实，但同时统一PYTHONHASHSEED/共同插桩，不能唯一归因旧分叉。
- init4在OFF-A/OFF-B/ON都600步，稳定长轨迹不能用观察器开关差异解释。接触/In只作诊断，不直接用作持稳/落稳奖励。下一步可CPU对齐这条轨迹的接触/位姿/原生In与RGB，定位返工/阶段停滞，仍不冒充物理因果证明。
- 自有12个client/server PID、supervisor/tmux daemon结束后均退出；随后另一已授权tile任务接手GPU，其工作与记录不计入本路线，不停止其进程。原始runtime/旧raw/protected保留，Knowledge增量同步最新共享记录。
- **下一步（未启动）**：单独设计同phase/同并发需求T/A_off成本及实际target工作账本，沿用可复现新起点合同；CPU查init4 600步返工；据此再决定冻结target的任务反馈/困难状态draft训练。任何新GPU/源码插桩/训练预算另确认，不自动加局、RL/KL或大小模型联训。

[结果报告](../../debug/Debug-016-speculative-repro-policy-six-results-20261011.md)；[CPU复核](../../../review/OMISSIONS.md)；[完整比较](../../../review/OMISSIONS.md)；[证据目录](../../../review/OMISSIONS.md)；[自动结果原件](../../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/FINAL_REPORT.md)。

## 19. 稳定600步的第二次抓取返工与成本对照设计（2026-10-11T13:07:01+08:00）

用户“可以的”后只CPU读取已有两条S0原生轨迹，不启动新GPU/服务/模拟器/训练，不改公共源码。第18节复现/开关关卡结论保留，当前进一步定位阶段，不改写旧失败。

| 标记/阶段 | init3 | init4 | init4−init3 |
|---|---:|---:|---:|
| 第一罐首次In | 168 | 167 | −1 |
| 第一In→第二罐首次双指contact | 69 | 359 | +290 |
| 第二罐首次双指contact→success | 56 | 74 | +18 |
| 总步数 | 293 | 600 | +307 |

- 两不同init的+307步中+290集中在第二次接近/建立抓取窗口（94.46%，仅描述性定位，不是K2相对T因果分解）。soup首次In后均一直真；In不等于释放落稳，双指contact不等于持稳。
- init4四闭合段242–299/342–367/392–425/478–489（58/26/34/12步）均无采样双指，抬升最高6.55mm。第三段仅2个单指样本；闭合前left232–241、389–393也存在，不说“之前没碰到”。第五523–591，526起双指且明显提起，600原生目标真。原始图像支持罐留桌面、手部反复靠近而未提起，不当搬运掉落或唯一根因证明。
- 失败段最小距离时EEF-site比罐中心高约41.0–49.5mm，成功段约7.1/8.3mm；关节slide差前段很小、成功段约63.8mm，是闭合前位姿/空合线索，不是指尖净空/力或已证深度/L2原因。每env.step后采样不能排除内部短接触。
- CPU独立对账895 native行/893执行命令/24组原双RGB及729去重文件hashPASS，旧runtime/source/raw/protected保留；图像无合成补帧，跨阶段chunk不重复求和作真实请求数。
- **成本设计已完成、实验未启动**：15静态/纯CPU队列合同通过；已有counter仅active profile有效，边界CUDA同步，T从phase.calls核对target，不能空counter=0，也不拿profile/sync E2E当clean收益。当前K2仍7逻辑/14 CFG行，bonus/3行优化不改。reset/cache内部profile缺失保持缺失；任务同步/边界包装未实现。
- **下一步建议（需另确认）**：先最多4完整episode T/S0×init3/4，统一新seed/首局/公共指纹/native协议，fresh S0需复现293/600，否则停关卡查第一分叉；判断第二罐返工是否相对T新增。随后另确认T/A_off成本包装及最多8局同步需求profile诊断，single-demand尾部完整保留、不补假请求/截断；profile-off自然E2E另关卡。未占用或停止并行tile任务，不自动追加。
- 训练思路据阶段定位聚焦闭合前关键位姿/困难状态，但此部署动作数据没有实际denoise predictor/同query teacher features，不能直接当draft训练样本。继续冻结target，L2局部辅助；同init反事实/标签合同确认后再定任务反馈监督，不自动RL/KL或大小模型联训。

[阶段报告](../../debug/Debug-017-speculative-init4-grasp-rework-cost-design-20261011.md)；[成本草案](COST-MATCHED-DESIGN-20261011.md)；[验证](../../../review/OMISSIONS.md)；[阶段数据](../../../review/OMISSIONS.md)；[原始图像](../../../review/OMISSIONS.md)。

## 20. 受控同init T/S0四局对照已授权、运行中（2026-10-11T13:18:44+08:00）

用户确认第19节下一关；不是重复询问方案，已实际启动。先S0双独立EGL首局init3/4→CPU全测轨迹复现旧ON（293/600步、19/38chunk、20初始数组、逐步state/RNG/RGB/command/原生物理字段、全部action/video）→通过才T两局。失败停止、不补局，不修改公共源码/target/verifier/阈值/bonus。

准备13 CPU+17包装合同与114张量参考自比较/故障注入PASS，240冻结输入hash/9target stat；服务器tmux 2026-10-11T13:16:48.534856+08:00启动、daemon PPID1/PTY stdin，端口19651/19652/29651/29652。当前模型结果待完成；native/指纹两组开，teacher/profile/full trace/训练/RL/联训关闭，不是clean timing。

本关回答同init第二罐返工是否相对T增加，并保留第一分叉；不把不同init +307步或几何线索当算法因果。完成后CPU复核并另增结果记录；成本T/A_off同步包装/诊断与困难阶段同query训练另授权，不自动追加。

[启动记录](CONTROLLED-T-S0-4EP-20261011.md)；[动态状态](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/status.json)；[自动结果](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/FINAL_REPORT.md)（完成/失败后生成）；[准备证据](../../../review/OMISSIONS.md)。

## 21. 受控同init T/S0四局结果与下一关（2026-10-11T13:37:58+08:00）

用户第20节授权任务已用尽4局，2026-10-11T13:16:48.534856+08:00→2026-10-11T13:34:48.593274+08:00，18.00分钟；4/4成功。S0先全测复现旧ON 293/600（逐步已测state/RNG/RGB/command/原生物理字段、20初始数组及114完整action/video张量），才运行T。T init3/4 293/300步，S0 293/600步；初始20数组/client RNG/RGB匹配。旧自动报告/旧runtime/raw/protected不改。

| init | S0−T总步数 | 第一In差 | 第二接近/建立双指窗口差 | 第二双指→成功差 |
|---|---:|---:|---:|---:|
| 3 | 0 | 13 | -13 | 0 |
| 4 | 300 | 8 | 280 | 12 |

本轮两对S0−T总步数合计+300；两init并非一致增加；必须逐init解释，不能将init4的600步概括成所有状态都退化。 详细两算法闭合段、contact/In区间、几何描述与第一分叉见报告；阶段记账不是去噪深度/动作维度/L2/z唯一因果。In仍是包含，contact不是稳定抓持，早期输出差异可通过闭环累积。

CPU独立对账95chunks/1486命令/1490指纹/1490native/1440原双RGB/4终止快照、95对应chunk完整action/video与676去重文件hashPASS；本turn9次轮询。无teacher/profile/full trace或新训练/RL/联训、无公共源码/target/verifier/阈值/bonus改动，与tile隔离。

**下一步**：另授权同负载T/A_off成本job-only包装与CPU合同，再单独确认最多8局profile/同步诊断；profile-off自然E2E另关卡。先核对真实target work/CFG/shape/request cohort和不配对完整尾部，不让profile诊断时间冒充部署收益。困难阶段同candidate-state冻结target监督训练后置，成功/步数/长尾/work/E2E主验收、L2辅助；本轮部署动作缺query训练特征，不能直接训练。小样本不证明正式非劣/泛化/完整hidden-state或旧失败唯一根因，不自动追加成本GPU/episode/训练。

[结果报告](../../debug/Debug-018-speculative-controlled-target-s0-four-results-20261011.md)；[CPU验证](../../../review/OMISSIONS.md)；[阶段数据](../../../review/OMISSIONS.md)；[第一分叉](../../../review/OMISSIONS.md)；[数据图](../../../review/OMISSIONS.md)；[原自动报告](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/FINAL_REPORT.md)。

## 22. 同负载T/A_off成本包装实现与CPU验证（2026-10-11T14:53:07+08:00）

用户明确授权“实现并CPU验证…记录HW…先不启动新的GPU任务或训练”。本輪仅新增独立任务目录，不改公共模型/server/client/target/verifier阈值/cache/bonus/protected；与tile隔离，旧runtime/raw/第21节和历史成本草案保留。

- **实现已完成**：[任务目录](../../../投机采样/experiments/cost_matched)与[README](../../../投机采样/experiments/cost_matched/README.md)含双worker loopback cohort、client/server逐请求账本、transformer实际CFG/input shape及有效history选择后的Q/K/V shape、完整single-demand尾部、默认禁用入口和离线对账/分层CLI。只增加元数据，不补假请求/截断/额外model调用；超时/错配/失败停止。
- **CPU PASS：89项（76新增+13既有）**，7.53秒。实际两个CPU进程/TCP、19/38/19/19/51/19等队列保留完整尾部；正式client fake-env OFF/ON均15步/相同请求。提取当前公开attention/infer/profile方法在CPU小fixture执行，输出/RNG/cache/profile合同通过，不载入完整WAM/真实LIBERO。旧T真实profile17行每行50普通action+1terminal+21video，空counters不作零work。732冻结hash/9target stat/248预备source校验PASS，CUDA未初始化。
- **工作单位分开**：普通K2七逻辑/十四CFG；接受深度0/1的fallback可用verifier已有target方向提交一步，并不代表再一次transformer；actual calls/rows、committed scheduler、terminal、顶层/嵌套phase分别对账。无实际sidecar保持null，reset/cache内部phase未知保持未知。
- **匹配不强行**：按worker/task/init/frame/明确物理阶段/每行QKV shape/planned demand/observed overlap分层；batch倍数和call数是结果。无标签/缺KV/无对侧保持unmatched；记录overlap值/比例，但server区间重叠不是GPU kernel重叠。CPU合成4policy无标签全unmatched，有显式标签2分层/4匹配，仅是合同样例，不是模型成本测量。
- **未完成/不声称**：未新GPU/模型服务/真实episode/teacher/训练/RL/联训；真实GPU数值/完整轨迹开关与实际同负载成本NOT_RUN。profile本身CUDA同步，新增Python记录也有开销；诊断不证明自然E2E收益、任务成功率非劣或隐藏输入完全相同。
- **下一步（另确认）**：先明确真实GPU完整输出/轨迹不扰动关卡的预算/各自冻结T与A_off基线，不能把旧A_off不同seed/首局历史直接当新合同参考。通过后再单独批准候选T→A_off→A_off→T各init3/4双EGL首局、最多8完整成本诊断；prepared合同GPU批准false、GPU关卡NOT_RUN，当前不可自动执行。profile-off自然E2E及困难阶段同candidate冻结target训练再后置；L2辅助，任务成功/步数/长尾/work/E2E主验收。

[实现报告](../../debug/Debug-019-speculative-cost-wrapper-cpu-20261011.md)；[CPU验证](../../../review/OMISSIONS.md)；[最终回归日志](../../../review/OMISSIONS.md)；[未授权预备计划](../../../review/OMISSIONS.md)。

## 23. 最小真实GPU不扰动验证预算规划（2026-10-11T15:07:45+08:00；待用户确认）

用户要求先给最小局数、时间与通过标准，确认预算后再启动，暂不训练，每步写HW并提醒下一步。本轮只CPU只读资料/规划与记录，不改新旧源码/原成本guard/protected，不新GPU或服务。

- **建议6新完整episode/最多6启动尝试**：T_ON2（init3/4双EGL首局）复用已受控T_OFF 293/300步/19+19chunks→全测通过才Aoff_OFF2→完整性通过才Aoff_ON2。两侧native/公共指纹同开；ON仅profile＋cost sidecar＋cohort，OFF全关。Aoff旧init3第二局且缺新seed/全测合同，必须新基线，不预定287/289步。
- **6局的条件**：旧T来源/seed/首局/公共插桩合同须核对并完整复现；否则停，不自动补成全新8局。只4局仅覆盖Aoff、漏掉T真实路径。全新T_OFF/ON＋Aoff_OFF/ON需8局，另确认，当前不授权。
- **时间/安全预算待确认**：GPU阶段30–50分钟，准备/CPU核验/HW合计45–75分钟，不含GPU排队/不保证。建议90分钟GPU运行阶段安全上限；到限failed/incomplete、保留数据，不把截短算通过/不重试。历史T约374秒/旧Aoff369–388秒只用于估时，不当新Aoff结果。
- **通过标准**：同初态20数组/client RNG/RGB/XML/init；每chunk完整30通道action/video shape/dtype和逐元素精确一致；逐步已测state-RNG-RGB-command/native/done/reward及终止success/horizon/steps/chunks全部相同；完整client/server request/cohort/实际CFG/QKV/target calls与scheduler counters账本符合，缺失不补0。单看success/步数或L2不够；不放宽容忍。未保存server RNG/全hidden cache不声称全等，profile/屏障耗时可以变，不是clean E2E。
- **输入检查已完成，GPU仍NOT_RUN**：761既有文件hash/248预备source/9target stat，T参考595行/76完整张量文件合同；上一关89 CPU PASS未本轮重跑。GPU资源可用性未查，不声称可启动。
- **确认后的第一步**：新增仅本任务6局监督器/通用CPU比較器，补自比较/故障注入/预算模式与CPU回归，再资源/EGL/provenance门。原8局成本入口必须保留“不扰动先PASS”guard，不能绕过；当前该新增工程未实施。各轮完成报告本步结果与下一关，失败停不补局。
- **最后仍需提醒下一步**：本关全PASS后，最多8局T/Aoff同负载成本诊断单独确认；profile-off无屏障自然E2E、困难阶段同query冻结target训练更后置。当前所有新GPU预算/代码实施待确认，不自动成本/teacher/训练/RL/KL训练/联训。

[6局完整方案](GPU-NONPERTURB-6EP-PLAN-20261011.md)；[规划输入检查](../../../review/OMISSIONS.md)；[参考资料](../../../review/OMISSIONS.md)；[历史估时证据](../../../review/OMISSIONS.md)；[未批准预算](../../../review/OMISSIONS.md)。

## 24. CPU预检完成（2026-10-11T15:29:24+08:00；PASS）

投机采样CPU预检完成：PASS；已预留0/6次尝试，GPU启动=false。 下一步重新核验GPU/端口/独占锁后，按预算启动T_ON两局。暂不训练。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：下一步重新核验GPU/端口/独占锁后，按预算启动T_ON两局。暂不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 25. GPU启动前关卡通过（2026-10-11T15:30:25+08:00；PASS）

投机采样GPU启动前关卡通过：PASS；已预留0/6次尝试，GPU启动=false。 资源门后启动T_ON两局，对比旧受控T_OFF；不训练。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：资源门后启动T_ON两局，对比旧受控T_OFF；不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 26. T_ON轮完成（2026-10-11T15:37:52+08:00；PASS）

投机采样T_ON轮完成：PASS；已预留2/6次尝试，GPU启动=true。init3 success=true，293步/19chunks，38完整张量，294行已测指纹；init4 success=true，300步/19chunks，38完整张量，301行已测指纹 下一步仅A_off_OFF两局建立同合同基线。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：下一步仅A_off_OFF两局建立同合同基线。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 27. A_off_OFF新基线轮完成（2026-10-11T15:46:19+08:00；PASS）

投机采样A_off_OFF新基线轮完成：PASS；已预留4/6次尝试，GPU启动=true。init3 success=true，283步/18chunks，36完整张量，284行已测指纹；init4 success=true，291步/19chunks，38完整张量，292行已测指纹 下一步仅A_off_ON两局，对比本轮OFF全部输出/已测轨迹。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：下一步仅A_off_ON两局，对比本轮OFF全部输出/已测轨迹。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 28. A_off_ON轮完成（2026-10-11T15:54:36+08:00；PASS）

投机采样A_off_ON轮完成：PASS；已预留6/6次尝试，GPU启动=true。init3 success=true，283步/18chunks，36完整张量，284行已测指纹；init4 success=true，291步/19chunks，38完整张量，292行已测指纹 本关结束；下一步CPU独立复核。8局成本/自然E2E/训练均须另确认。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：本关结束；下一步CPU独立复核。8局成本/自然E2E/训练均须另确认。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 29. 6局总验收完成（2026-10-11T15:54:37+08:00；PASS）

投机采样6局总验收完成：PASS；已预留6/6次尝试，GPU启动=true。 独立CPU复核；再单独确认最多8局同负载成本预算。不自动GPU、teacher或训练。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：独立CPU复核；再单独确认最多8局同负载成本预算。不自动GPU、teacher或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

## 30. CPU二次复核与GPU收尾（2026-10-11T15:56:53+08:00；PASS）

投机采样CPU二次复核与GPU收尾：PASS；已预留6/6次尝试，GPU启动=true。 本轮二次CPU复核及收尾完成；下一步先CPU汇总已有T/A_off成本账本的阶段/匹配覆盖，再单独确认最多8局同负载成本实验预算。自然E2E及冻结target训练后置，不自动GPU或训练。 暂不teacher/训练/RL/联训，非tile，不是clean E2E或正式非劣证明。

用户已确认6次尝试上限＋90分钟GPU安全限时＋任务专用监督器/CPU比较器。旧第23节“待确认”是历史规划状态，未篡改为当时已批准。每轮两独立EGL首局init3/4；严格全输出/已测轨迹及ON请求/计数/QKV账本，不重试/补局，暂不训练。

下一步：本轮二次CPU复核及收尾完成；下一步先CPU汇总已有T/A_off成本账本的阶段/匹配覆盖，再单独确认最多8局同负载成本实验预算。自然E2E及冻结target训练后置，不自动GPU或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始证据](../../../review/OMISSIONS.md)；[job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)。

本关完整汇总：[6局结果与边界](GPU-NONPERTURB-SIX-RESULT-20261011.md)。当前总览行已同步为GPU不扰动PASS；不改变历史第23节待确认规划或既有原始记录。

## 31. 现有T/A_off成本账本与原生阶段匹配CPU整理（2026-10-11T16:27:39+08:00；PASS）

本轮用户只授权CPU整理已有资料、写HW并提醒下一步；无新GPU/服务/episode/teacher/训练/RL/联训，无公共/成本包装/权重/阈值/cache/bonus/protected修改，与tile隔离。

- **覆盖**：150边界（4reset/75policy/71cache）；T38/Aoff37policy。实际chunk锚点首chunk12步；以请求开始当前In与双指接触标阶段。同worker/init/frame、物理阶段、每行shape/dtype/QKV、双需求/观测重叠/provenance匹配32对/64请求（85.33%）。11保留未匹配：8物理阶段不同、1对側提前结束、2需求/重叠尾部不同；不强凑。
- **真实账本差异**：32对action7.759→9.771秒（+25.93%），server policy11.345→13.456（+18.60%）；10对快/22对慢，两worker一致。action网络增加0.517秒、其余增加1.495秒；构树0.995含draft0.318，嵌套timer不重复加。action调用51→30.41却CFG行102→393.44（3.86倍、不是FLOPs）；屏障5.24→7.20毫秒非直接主因，video3.519→3.612秒，reset大固定成本内部未知。
- **短/长段不同**：短有效KV144–720（10对）action−10.14%；增长864–2016（16对）+24.81%；饱和2160（6对）+82.83%。verifier每调用0.186→0.401秒，depth2/tree71.97%→34.03%，draft提交82%→54.33%；不是预测在线阈值。当前Aoff环境总步少19却只少1总chunk，不能把成本退化全归环境步数。
- **证据边界**：32对中仅2对chunk0已测初态全同、独立init仅2个，不是同query数值因果/自然E2E/非劣或kernel重叠；阶段/KV/任务难度/并发尚不能唯一分开。细化接触/command仍32对，去跨阶段25对action仍+22.45%。Aoff单需求长KV尾部action5.611秒不混入双需求，不作独占GPU因果证明。
- **CPU验证**：19项独立数据/代数/故障注入检查PASS，stdlib未import Torch/模型/模拟器；1532旧SHA/9target stat保护。初版测试fixture漏chunk的错误日志保留，只修临时fixture后通过；实际读数未变、历史132项未重跑。
- **下一步**：先CPU规划短/长KV×单/双需求固定负载最小成本对照，明确同输入/形状/计时合同与GPU预算；确认后才执行。原8局ABBA需据长KV/并发问题修订，不自动GPU或训练；自然E2E及冻结target困难状态训练/RL/联训后置。 接受率/trace长度单项改善不能保证成本回本；暂不RL/KL或大小模型协同训练。

[完整读数报告](COST-LEDGER-CPU-READOUT-20261011.md)；[分层账本](../../../review/OMISSIONS.md)；[32对CSV](../../../review/OMISSIONS.md)；[CPU验证](../../../review/OMISSIONS.md)；Knowledge CHAT-20261011-SPECULATIVE-COST-LEDGER-CPU-READOUT。

## 32. 短/长KV×单/双需求最小固定负载成本方案（2026-10-11T16:58:36+08:00；预算待确认）

本轮仅CPU规划/已有产物核查与HW记录，未新GPU/模型服务/EGL/episode/工程实现/teacher/训练/RL/联训。公共模型/成本包装/旧8局guard/protected、旧证据与tile条目保持。

- **最小局数**：建议0新完整LIBERO/模型闭环episode，最多2次双EGL init3/4初态初始化（各5稳定步、policy env动作0）。旧T_ON保存obs_data/actions/latents，prefix两worker84文件CPU读取、440 raw双相机帧有限/hash；初态重建或cache关失败就停，不自动补新局。0完整局不等于0GPU。
- **四格负载**：chunk1/frame4/KV288（避首帧零条件）与chunk14/frame56/KV2160，各单/双需求；2历史init共4fixture。两模型驻留，single peer空闲、dual两worker同fixture/算法/RNG，避免不同接受深度尾部。T/A固定同历史/噪声起点，但不要求T/A action相同；同算法跨恢复/包装/profile/单双需求须全输出、有效cache/RNG精确复现。
- **数量/时间待确认**：主测ABBA+BAAB两块，每算法每fixture每需求4次；64primary+32peer=96主测、48warmup+24诊断，总168完整固定chunk调用上限（不是168局）。GPU估35–55分钟、建议首个owned GPU/EGL/model进程起90分钟硬cap；CPU工程45–90分钟/复核10–20，总90–165不含排队、非保证。不重试/加第三块/自动8局或训练。
- **通过标准**：先新增仅任务专用默认关闭snapshot/replay/监督器/独立CPU比较器并G0测试；资源/EGL/20初态数组-RNG-RGB/30block实际KV及原slot合同；同算法完整30action/video/有效cache/RNG/work精确；primary profile OFF仅整体边界同步，ON另分解、嵌套不加重；双action overlap≥90%、4次MAD/median≤7%（均是待确认测量门），超过不自动扩样。门等待action外/policy内单列，非自然E2E。
- **判读不等于完整性**：r=medianA/medianT，<0.95或>1.05且两init/两块同向才作pilot加速/退化信号，±5%内持平、混合/噪声不确定；慢是有效结果，不自动改树宽/阈值/训练。短/长frame/history不同，不能唯一归KV；peer和重复非独立episode，非任务非劣或GPU kernel证明。
- **CPU输入关PASS，GPU NOT_RUN**：8规划合同检查（非新G0工程测试）、1559旧SHA/9target stat不变，CUDA CPU映射未初始化，不导入模型/模拟器。旧全KV/server RNG未保存，GPU恢复尚未证明；归一化动作须原postprocess/preprocess路径或先证shortcut等价，原slot/容量不compact，无效torch.empty槽不要求finite/全字节同。初版raw帧/空间shape假设错误的两日志保留，仅修临时审计后通过。
- **下一步**：等确认0新完整episode＋最多2次EGL初态初始化＋168固定chunk调用＋90分钟GPU阶段上限和仅任务专用默认关闭snapshot/replay/监督器/CPU比较器实现范围。先CPU工程预检，再新鲜资源/EGL门、fixture恢复不扰动门和四格成本；失败不补局/加块/改容忍，不自动旧8局、自然E2E或训练。 自然E2E、冻结target真实状态训练/RL/KL/联训继续后置。

[完整固定负载方案](FIXED-LOAD-COST-PLAN-20261011.md)；[未批准预算](../../../review/OMISSIONS.md)；[产物inventory](../../../review/OMISSIONS.md)；[CPU规划输入关](../../../review/OMISSIONS.md)；[8合同测试](../../../review/OMISSIONS.md)；Knowledge CHAT-20261011-SPECULATIVE-FIXED-LOAD-COST-PLAN-PENDING。

## 33. 进展检查点、WA-SpecDec与相邻文献（2026-10-11T17:32:50+08:00；建议待确认）

- **先保存进展**：[检查点](RESEARCH-CHECKPOINT-BEFORE-WASPECDEC-20261011.md)整理机制打通、历史单chunk收益/任务步数问题、20局冻结target A_off/B_on、种子/观察器复现、最新六局T/A_off不扰动和成本账本。最新T293/300、A283/291步，步数总少19但matched action+25.93%；短−10.14%/长KV2160+82.83%，质量与成本分开。
- **调研核验**：[完整报告](WASPECDEC-LITERATURE-REVIEW-20261011.md)、[19项书目](../../../review/OMISSIONS.md)和[说法审计](../../../review/OMISSIONS.md)。14正式主会论文集核验；KERV另为作者报告DAC2026接收、未核验DOI；WA/HeiSD/Dex/FFDC未核验正式接收，不等于拒稿。按2026第七版CCF，ICLR/ICML/NeurIPS/ASPLOS/DAC A、EMNLP B；当前评级与历史/机构政策分开。
- **WA借鉴与边界**：世界条件与训练方法值得研究，但它先改target再冻结蒸馏，非RL或原target冻结全程；不是只有一个模拟基准。固定rho消融增量约11.7%与同成功率1.5×口径不同，不能当完整任务加速。距离验收不是严格分布保持，NCF不是无条件碰撞风险，不据“简单/预印本”判无价值。
- **本地新观察**：训练decode与部署predict输入动作去噪latent、上一方向、sigma/next_sigma/timestep，无显式image/world/target hidden；上一方向间接含场景，不能说完全无条件。建议优先复用冻结WAM已有特征并计读取/offload成本，收益尚未验证，不立刻新增世界encoder或LoRA。
- **方案建议，未获批实施**：不推倒重构；保留冻结原T/原cache/torch/阈值，先成本关，再成本感知启停/预算及世界条件、候选自身访问状态监督训练。成功/步数/返工/失败尾部/自然E2E共同验收，KL不保证trace，RL与大小权重联训后置；已存在相关文献，不宣称“首次WAM+投机”或现有实现更优。
- **本轮边界**：只文献/CPU读源与HW记录，不新GPU/EGL/服务/episode/teacher/训练/RL/联训、源码或保护文件修改；初始1623文件中1620不变、3个外部tile文档更新保留，最新写入前1628SHA/9target stat写后核验，不回滚并发更新、不混入tile研究。
- **下一步**：先确认“不重构、成本优先、世界条件训练后置”的研究建议。若继续原固定负载关，还需明确批准0新完整episode、最多2次EGL初态、168固定chunk和90分钟GPU硬cap及任务专用默认关闭实现范围；先CPU工程门，再资源/fixture恢复/成本。失败不补局、不自动改阈值/自然E2E或训练；RL/联训后置。 原固定负载方案的G0/资源/精确恢复/负载完整性门均保持；当前预算待确认、GPU NOT_RUN，不把调研建议当授权。

Knowledge CHAT-20261011-SPECULATIVE-WASPECDEC-LITERATURE-REVIEW；[写后保留检查](../../../review/OMISSIONS.md)。

## 34. 预算与实现授权已记录（2026-10-11T17:49:58+08:00；AUTHORIZED）

投机采样固定负载预算与实现授权已记录：AUTHORIZED；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；补齐任务专用默认关闭包装、独立CPU比较器和安全监督器；先完成CPU工程门，再GPU资源与恢复关。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：补齐任务专用默认关闭包装、独立CPU比较器和安全监督器；先完成CPU工程门，再GPU资源与恢复关。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/plan.json)。

## 35. CPU工程预检完成（2026-10-11T18:09:34+08:00；PASS）

投机采样固定负载CPU工程预检完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；CPU工程门通过；下一步新鲜独占资源/端口检查，然后启动两模型与两EGL初态，进入fixture恢复门，仍不训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：CPU工程门通过；下一步新鲜独占资源/端口检查，然后启动两模型与两EGL初态，进入fixture恢复门，仍不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/plan.json)。

## 36. GPU启动前资源门完成（2026-10-11T18:13:39+08:00；PASS）

投机采样固定负载GPU启动前资源门完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/plan.json)。

## 37. 关卡失败停止（2026-10-11T18:14:12+08:00；FAIL）

投机采样固定负载关卡失败停止：FAIL；GPU启动=true，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

失败关卡：RuntimeError("owned process exited before ready ['/rjs/cxz/miniconda3/envs/va/bin/python', '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port=29861', '--local-ranks-filter=0', '--tee=3', '/rjs/cxz/论文探索/投机采样/experiments/fixed_load_cost/worker_entry.py', '--job', '/rjs/cxz/论文探索/投机采样/results/runtime/p0-fixed-load-cost-20261011-174601', '--worker', '0']")

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/plan.json)。

## 38. 固定负载首次启动失败后CPU复核与下一步（2026-10-11T18:20:36+08:00）

投机采样固定负载第一次启动FAIL并已清理：195项CPU工程测试通过，但新Runtime在默认off、fresh predictor下请求原受保护draft loader，两模型均拒绝。CPU抽取真实公共loader/完整wrapper AST复现；原fake预加载predictor/lambda绕过保护，定位预检盲点。新完整episode0、EGL0/2、固定chunk0/168，未到恢复/G1/成本格；失败快照约32.06秒、GPU显存归零。1128旧SHA、11实现/plan SHA与9target stat不变；无修复/重启/训练，非tile，不产生新成本或质量结论。

本次是任务包装初始化错误，不是显存/模型质量/成本退化。GPU FAIL保持；CPU诊断PASS只证明guard矛盾，不证明修复或恢复关。历史195 CPU PASS和资源PASS、旧成本/六局结论保留。

建议仅任务专用未计时初始化区try/finally临时verifier_k2_b2预加载真实A_off并恢复off，补真实loader/fresh-cached/成功-异常恢复的CPU回归。建议未实施，不改11文件与plan、不重启失败one-shot；新GPU需另确认。

下一步：先确认仅任务专用预加载顺序修复与真实loader CPU回归；CPU通过后再单独确认新GPU job，不恢复失败one-shot，不训练。

[汇总报告](FIXED-LOAD-COST-ATTEMPT-20261011-174601.md)；[CPU诊断](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/CPU-FAILURE-READOUT.json)；[原失败](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/FAILED.json)；[清理](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/OWNERSHIP-CLEANUP.json)；Knowledge CHAT-P0-FIXED-LOAD-COST-20261011-174601-POST-CPU。

## 39. 任务预加载顺序CPU修复与真实接口回归（2026-10-11T19:58:28+08:00；CPU PASS，GPU NOT_RUN）

投机采样任务预加载CPU修复通过：仅worker/test两文件，初始化临时verifier_k2_b2并try/finally恢复原off，原公共保护不改。新增14真实loader/actual A_off/首次-缓存/加载-迁移-eval异常回归，固定负载77、合并209项通过；原va环境直接真实server接口CPU预加载通过，112维/参数一致/eval/CUDA未初始化。1175旧SHA、旧失败job33文件与9target stat不变，11旧实现归档；旧GPU gate因代码SHA变化拒绝复用，原FAIL保留。0新GPU/EGL/episode/chunk，无训练、非tile；恢复/不扰动/成本仍NOT_RUN，GPU另确认。

只修改任务worker/test；原constructor除预加载外和其余Runtime方法AST保持。原fresh/off接口保护不移除，真实A_off参数一致。模式恢复不承诺predictor/RNG全回滚；原loader缓存和fixture后置seed保留，异常不重试。

旧第38节“修复待确认”为历史，本次授权只完成CPU修复；旧GPU job仍FAIL、33文件不变，不覆盖gate，11旧实现原SHA归档。两个测试环境均隐藏CUDA，原va探针只用真实接口与最小CPU上下文，不加载/推理target。

下一步：单独确认新GPU job与0完整episode/2EGL/168chunk/90分钟预算后，重新冻结实现、完整CPU与资源门，再初态→fixture恢复→G1→四格成本；本轮不启动、不恢复旧失败job、不训练。

[完整报告](FIXED-LOAD-PRELOAD-CPU-FIX-20261011.md)；[验证](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/CPU-REPAIR-VALIDATION.json)；[209项实际日志](../../../review/OMISSIONS.md)；[原环境真实接口CPU探针](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/real-module-CPU-result.json)；Knowledge CHAT-20261011-SPECULATIVE-FIXED-LOAD-PRELOAD-CPU-FIX。

## 40. 预算与实现授权已记录（2026-10-11T20:04:36+08:00；AUTHORIZED）

投机采样固定负载预算与实现授权已记录：AUTHORIZED；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；新job已建立并冻结修复实现/旧输入；先209项CPU回归、原va真实接口CPU探针及84产物/440帧工程输入门，再资源与GPU恢复；失败停不补跑，无训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：新job已建立并冻结修复实现/旧输入；先209项CPU回归、原va真实接口CPU探针及84产物/440帧工程输入门，再资源与GPU恢复；失败停不补跑，无训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/plan.json)。

## 41. CPU工程预检完成（2026-10-11T20:06:27+08:00；PASS）

投机采样固定负载CPU工程预检完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；CPU工程门通过；下一步新鲜独占资源/端口检查，然后启动两模型与两EGL初态，进入fixture恢复门，仍不训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：CPU工程门通过；下一步新鲜独占资源/端口检查，然后启动两模型与两EGL初态，进入fixture恢复门，仍不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/plan.json)。

## 42. GPU启动前资源门完成（2026-10-11T20:07:04+08:00；PASS）

投机采样固定负载GPU启动前资源门完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/plan.json)。

## 43. 关卡失败停止（2026-10-11T20:07:55+08:00；FAIL）

投机采样固定负载关卡失败停止：FAIL；GPU启动=true，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

失败关卡：RuntimeError("owned process exited before ready ['/rjs/cxz/miniconda3/envs/va/bin/python', '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port=29872', '--local-ranks-filter=0', '--tee=3', '/rjs/cxz/论文探索/投机采样/experiments/fixed_load_cost/worker_entry.py', '--job', '/rjs/cxz/论文探索/投机采样/results/runtime/p0-fixed-load-cost-20261011-200434', '--worker', '1']")

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/plan.json)。

## 44. 新GPU启动失败后的CPU生命周期定位（2026-10-11T20:21:11+08:00；FAIL保留，仅定位）

投机采样新固定负载job在两模型启动失败停止：209 CPU回归通过但探针人工补action_per_frame，遗漏原reset生命周期；GPU约50秒，0完整episode/0EGL/0固定chunk。CPU真实接口复现缺字段，原reset后实际A_off加载通过（target cache明确CPU stub），只定位未修复。恢复/G1/成本NOT_RUN，1216冻结SHA/11实现/9target stat不变。本任务清理PASS；之后另一用户占GPU，不干预。下一步先确认CPU生命周期修复，GPU再确认，非tile。

原server constructor不设action_per_frame，原_reset:2121才建立；任务预加载在constructor:47，reset在build:133。本次实际模型两个worker同样失败，50.12秒错误快照/约50.90秒清理后状态；GPU安全上限未触发。前次CPU“真实loader/actual A_off通过”是最小上下文的局部结论，人工预填几何字段使其不覆盖新启动生命周期，不能拿209测试数当整套启动保证。

后验CPU隐藏CUDA，冻结Runtime复现缺字段且off恢复/no hooks；原reset(None)+明确CPU cache stub后真实A_off加载112维、参数一致。此检查不构造target、不执行prefix，不升级旧gate/FAIL，不声称GPU恢复或成本通过。1216输入/旧证据SHA、11实现、plan/A_off与9target stat不变。AUTH/CPU/START/FAILED分别已写HW。

本job所属进程/端口/tmux清理通过。20:16资源复查GPU不空闲；只读核实hlj另一进程20:15:43启动、非本job所有，不干预。下一次新GPU还需重过独占/空闲资源门。

下一步：用户确认后只做任务包装reset生命周期CPU修复/回归：原reset后、prefix/fixture seed和计时前预加载，不手填reset字段，补新启动→reset→load边界审计；完整CPU通过后新GPU job/预算及空闲资源另确认。两失败job不恢复、不补跑、不训练。

[完整报告](FIXED-LOAD-COST-ATTEMPT-20261011-200434.md)；[CPU诊断](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/CPU-LIFECYCLE-DIAGNOSIS.json)；[GPU失败](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/FAILED.json)；Knowledge CHAT-P0-FIXED-LOAD-COST-20261011-200434-POST-CPU。

## 45. CPU生命周期修复通过与成本测量职责澄清（2026-10-11T20:49:53+08:00；仅CPU PASS）

投机采样启动生命周期CPU修复PASS：仅worker/test，draft移到原reset后、prefix/fixture seed/计时前；bootstrap与draft就绪分离、build不重试。新增12生命周期回归，89固定/221合并通过；原va直接真实constructor/reset/dispatch→真实A_off→首prefix边界CPU probe通过，未手填reset字段，重组件明确stub，非GPU证明。1271基线除两获准实现外SHA不变、9target stat不变。CPU预检/比较器不测速模型，性能仍GPU实测。下一步授权新GPU关卡、暂不训练、非tile。

不再在constructor预加载、不手补reset字段；原构造→Runtime bootstrap→原reset→真实A_off→初态dispatch→首prefix边界CPU验证通过。明确target/VAE/text/tokenizer/cache/编码/FSDP为CPU重组件替身，未检验真实GPU完整prefix、恢复、G1或性能。旧代码同一fresh状态复现缺字段，7异常均不冒称draft/fixture就绪，重复build禁止。

T/A_off的正式成本是在A100上原target vs draft+target路径；CPU预检与读GPU日志的比较器不是CPU模型测速。GPU同步前后CPU计时反映实际路径wall time而非单kernel；额外hash/save在测量之后。以前CPU账本只是既有GPU证据的分析。

下一步：用户本次已确认继续：新建GPU job，重新冻结修复后实现/CPU gate，空闲资源通过后原reset/draft/prefix/恢复→G1→四格GPU成本；仍0episode/最多2EGL/168chunk/90min，不恢复旧失败job、不补跑、不训练，每关HW记录。

[完整报告](FIXED-LOAD-LIFECYCLE-CPU-FIX-20261011.md)；[实际constructor probe](../../../投机采样/results/runtime/p0-fixed-load-lifecycle-cpu-fix-20261011-203648/real-constructor-CPU-result.json)；Knowledge CHAT-20261011-SPECULATIVE-FIXED-LOAD-LIFECYCLE-CPU-FIX。

## 46. 预算与实现授权已记录（2026-10-11T20:53:19+08:00；PASS）

投机采样固定负载预算与实现授权已记录：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；用户已确认继续，CPU生命周期修复/221回归及原constructor probe通过；新job先重新实际CPU gate与资源门，再GPU恢复→G1→成本，失败停、无训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：用户已确认继续，CPU生命周期修复/221回归及原constructor probe通过；新job先重新实际CPU gate与资源门，再GPU恢复→G1→成本，失败停、无训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-205205/plan.json)。

## 47. CPU工程预检完成（2026-10-11T20:53:52+08:00；PASS）

投机采样固定负载CPU工程预检完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；新job实际221项CPU/原constructor-reset-draft-prefix入口probe与冻结工程门通过；下一步新鲜GPU空闲/端口资源门，真实两模型与两EGL，原reset/draft/prefix恢复→G1→四格GPU成本，无训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：新job实际221项CPU/原constructor-reset-draft-prefix入口probe与冻结工程门通过；下一步新鲜GPU空闲/端口资源门，真实两模型与两EGL，原reset/draft/prefix恢复→G1→四格GPU成本，无训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-205205/plan.json)。

## 48. GPU启动前资源门完成（2026-10-11T21:10:36+08:00；PASS）

投机采样固定负载GPU启动前资源门完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-205205/plan.json)。

## 49. 关卡失败停止（2026-10-11T21:11:04+08:00；FAIL）

投机采样固定负载关卡失败停止：FAIL；GPU启动=true，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

失败关卡：RuntimeError('stop/deadline requested')

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：已停止，不重试/补调用/初始化或训练；先CPU定位失败关卡，修复及是否再次GPU需用户另确认。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-205205/plan.json)。

## 50. GPU私有归属误停根因与CPU修复（2026-10-11T21:26:49+08:00）

投机采样新GPU job资源门通过但加载阶段被私有PGID监控误停：两个torchrun rank各自新建session，已由父launcher日志确认，非外部抢占。26.05秒触发/28.40秒清理记录，0EGL/0固定chunk/0完整episode，成本未测。新CPU私有监控改按父子链+root PID启动tick/UID/argv认领，原Torch constructor/_popen AST真实子进程拓扑及26回归通过，同用户无关进程仍拒绝。1347冻结文件/11原实现/9target stat不变，GPU不重跑、不训练，非tile。

以前第49节generic stop与原guard的foreign标签保留为当时原始症状，此处根据父launcher日志/原Torch代码/真实CPU拓扑纠正为本任务rank误判，不能归咎其他用户GPU抢占。本次GPU成本未测，221 CPU生命周期关不冒充GPU完整启动通过。

下一步：按失败停：不恢复/修改旧job，不自动GPU重试或训练。CPU私有监控v2已验证；若用户另确认新job与0完整episode/2EGL/168固定chunk/90分钟预算，重新预留GPU窗口和冻结CPU gate后，真实启动→两EGL初态→fixture恢复→G1→四格GPU成本。

[尝试报告](FIXED-LOAD-COST-ATTEMPT-20261011-205205.md)；[CPU修复](FIXED-LOAD-OWNERSHIP-CPU-FIX-20261011.md)；[debug](../../debug/20261011-fixed-load-ownership-guard.md)；Knowledge CHAT-20261011-SPECULATIVE-FIXED-LOAD-OWNERSHIP-CPU-FIX。

## 51. 预算与实现授权已记录（2026-10-11T21:30:38+08:00；PASS_AUTHORIZED_CPU_FIRST_NEW_JOB）

投机采样固定负载预算与实现授权已记录：PASS_AUTHORIZED_CPU_FIRST_NEW_JOB；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；新job先221工程回归/原constructor-reset probe/26私有ownership回归及完整冻结门；通过再资源门，真实GPU启动→两EGL初态→恢复→G1→四格成本。原0episode/2EGL/168chunk/90min，无训练，失败停、不自动重试。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：新job先221工程回归/原constructor-reset probe/26私有ownership回归及完整冻结门；通过再资源门，真实GPU启动→两EGL初态→恢复→G1→四格成本。原0episode/2EGL/168chunk/90min，无训练，失败停、不自动重试。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 52. CPU工程预检完成（2026-10-11T21:36:38+08:00；PASS）

投机采样固定负载CPU工程预检完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；新job 221项工程回归、26项私有ownership回归、原constructor-reset-draft-prefix入口probe与1380冻结工程门通过。CPU探针漏checkpoint env及一次CLI参数错误均在GPU前停止/更正并保留日志，未改源码。下一步独占资源门，真实GPU启动→两EGL初态→恢复→G1→四格成本；不训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：新job 221项工程回归、26项私有ownership回归、原constructor-reset-draft-prefix入口probe与1380冻结工程门通过。CPU探针漏checkpoint env及一次CLI参数错误均在GPU前停止/更正并保留日志，未改源码。下一步独占资源门，真实GPU启动→两EGL初态→恢复→G1→四格成本；不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 53. GPU启动前资源门完成（2026-10-11T21:45:36+08:00；PASS）

投机采样固定负载GPU启动前资源门完成：PASS；GPU启动=false，EGL初始化0/2、固定chunk调用0/168，新完整episode=0；启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：启动两固定模型，90分钟GPU限时开始；仅两EGL初态、先恢复门，无新完整episode。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 54. 两EGL初态门完成（2026-10-11T21:46:36+08:00；PASS）

投机采样固定负载两EGL初态门完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用0/168，新完整episode=0；释放初态renderer，不执行policy环境动作；用旧真实prefix建立四份固定fixture，再验全block/cache/RNG恢复。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：释放初态renderer，不执行policy环境动作；用旧真实prefix建立四份固定fixture，再验全block/cache/RNG恢复。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 55. 四fixture构建及恢复门完成（2026-10-11T21:50:25+08:00；PASS）

投机采样固定负载四fixture构建及恢复门完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用0/168，新完整episode=0；按48参考/包装warmup＋24诊断调用验证恢复、原路径/包装、OFF/ON及单双需求；未通过不进入主测。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：按48参考/包装warmup＋24诊断调用验证恢复、原路径/包装、OFF/ON及单双需求；未通过不进入主测。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 56. 恢复/包装/诊断不扰动门完成（2026-10-11T22:18:18+08:00；PASS）

投机采样固定负载恢复/包装/诊断不扰动门完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用72/168，新完整episode=0；进入S1→L2→L1→S2主测；每格反序两块，固定调用额度，失败停，不训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：进入S1→L2→L1→S2主测；每格反序两块，固定调用额度，失败停，不训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 57. 短历史单需求主测完成（2026-10-11T22:21:00+08:00；PASS）

投机采样固定负载短历史单需求主测完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用88/168，新完整episode=0；继续下一预定成本格；慢结果不是失败、不自动优化或训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：继续下一预定成本格；慢结果不是失败、不自动优化或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 58. 长历史双需求主测完成（2026-10-11T22:35:48+08:00；PASS）

投机采样固定负载长历史双需求主测完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用120/168，新完整episode=0；继续下一预定成本格；慢结果不是失败、不自动优化或训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：继续下一预定成本格；慢结果不是失败、不自动优化或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 59. Realtime-VLA FLASH/Fast-WAM适用范围与发表路线审查（2026-10-11T22:38:25+08:00）

用户澄清FastVLA指Realtime-VLA FLASH，非同名视觉压缩/训练库。action chunk并行与去噪单步是不同轴：当前LingBot同样整块16动作/50去噪更新；Fast-WAM视觉单次编码但动作10步，π0/0.5代码默认10步、π0.7论文5步，OFT为并行回归。FLASH在π0直接draft动作块/旧prefix KV并行端点检查、动作前缀执行与阶段回退；本路线draft去噪transition/target验收不是同层，均非无损轨迹保证。发表目标应覆盖现代轻动作专家与强优化基线，不能只换模型或宣称首个连续投机。GPU成本既有任务继续，文献不新增GPU/训练，非tile。

用户澄清论文身份并询问CCF A，未要求停止当前GPU或授权重构。当前GPU阶段头保持原执行状态，不用文献完成冒充成本完成。公开代表范式不是市场占比排名；action horizon与denoising steps分开。方法新意需超越已有FLASH与ICML连续diffusion投机；单纯模型换名、阶段fallback或放宽L2均不足。

下一步：先完成本次已授权固定成本GPU矩阵/终审与清理，不改当前模型、阈值或预算，不训练。随后仅CPU汇总成本/工作账本并制定跨范式可迁移性与新意检查：Fast-WAM/π类5–10步动作专家、FLASH对照、任务成功/环境步数/长尾/E2E联合合同。迁移实现、新GPU或训练均另确认，不因讨论自动重构。

[详细原始来源/方法边界/建议](REALTIME-VLA-FLASH-FASTWAM-SCOPE-20261011.md)；Knowledge CHAT-20261011-SPECULATIVE-FLASH-FASTWAM-SCOPE。

## 60. 长历史单需求主测完成（2026-10-11T22:48:21+08:00；PASS）

投机采样固定负载长历史单需求主测完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用136/168，新完整episode=0；继续下一预定成本格；慢结果不是失败、不自动优化或训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：继续下一预定成本格；慢结果不是失败、不自动优化或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 61. 短历史双需求主测完成（2026-10-11T22:52:35+08:00；PASS）

投机采样固定负载短历史双需求主测完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用168/168，新完整episode=0；继续下一预定成本格；慢结果不是失败、不自动优化或训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：继续下一预定成本格；慢结果不是失败、不自动优化或训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 62. 固定负载成本总复核完成（2026-10-11T22:54:15+08:00；PASS）

投机采样固定负载固定负载成本总复核完成：PASS；GPU启动=true，EGL初始化2/2、固定chunk调用168/168，新完整episode=0；先根据四格成本及ON拆解规划下一最小实验，任何缩树/启停/自然E2E或世界条件训练均另确认；不自动训练。 冻结原target/A_off/阈值，无训练/RL/联训，非tile、非自然任务E2E。

批准执行原0完整局/2EGL/168chunk/90分钟成本关，保持G0→资源→恢复/不扰动→四格成本；旧第32/33节待确认是历史状态，不改写当时记录。不重构、不训练、不改target/阈值/公共源或tile。

下一步：先根据四格成本及ON拆解规划下一最小实验，任何缩树/启停/自然E2E或世界条件训练均另确认；不自动训练。

[本轮报告](../../../review/OMISSIONS.md)；[原始结果](../../../review/OMISSIONS.md)；[已批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)。

## 63. 固定成本完成后的CPU终审、真实收益边界与发表路线（2026-10-11T23:11:34+08:00）

投机采样固定GPU成本已完成：0完整episode、2EGL、168chunk、68分38秒，全部有效性关/CPU终审通过并清理。短单/双动作耗时降约31%/15%；长单慢4.7–11.2%、长双慢28.2–30.2%。调用少但CFG行量约3.6–4倍；未证明质量非劣、环境步数或E2E。下一步建议仅CPU审查verifier成本与Fast-WAM/π迁移新意，不追加GPU/训练，非tile。

正式OFF动作分支：S1 −30.9%至−31.3%，S2 −14.9%至−15.3%；L1 +4.7%至+11.2%（按预注册为flat_or_mixed，不称加速），L2 +28.2%至+30.2%。T 51前向/102CFG行调用，A_off 29–32/370–412；主测单双需求同fixture工作完全一致。ON verifier约69–84%、draft约2–5%仅瓶颈线索，保留ON异常且不混OFF。

本次实际CUDA原reset→A_off预加载、全部恢复/不扰动/成本关、清理及1380文件/9target stat事后CPU审查通过；旧失败不覆盖。研究现代chunk范式是扩大验证而非否定现有方法：LingBot也整块输出，Fast-WAM视觉single-pass但动作10步，FLASH已经做动作块投机，不能宣称首个连续投机或把换模型当新意。

下一步：建议先仅CPU核对真实K2B2循环的tree/CFG行量边界与compact verifier/工作复用/负载准入候选，再规划现代Fast-WAM或π类5–10步迁移、强基线及success/环境步数/长尾/E2E合同；实现、新GPU及训练/RL/KL/联训另确认。

[成本全表、范围和候选关卡](FIXED-LOAD-COST-RESULT-20261011-213036.md)；[文献原始来源](REALTIME-VLA-FLASH-FASTWAM-SCOPE-20261011.md)；[实际最终结果](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/FINAL_RESULT.json)；[CPU终审](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/CPU-final-review/AUDIT.json)；Knowledge CHAT-20261011-SPECULATIVE-FIXED-LOAD-COST-FINAL-REVIEW。

## 64. verifier系统优化假设、Fast-WAM方向投机与双需求释义（2026-10-11T23:30:28+08:00）

投机采样CPU解释复核：双需求是同一A100两独立worker同算法/fixture并发，非B2/CFG/大小模型协同；长单已无净收益。verifier每层index_select将2份history物化14份再cat，有nonzero/CPU标量同步线索，根因未隔离。mega执行/共享KV与Fast-WAM方向投机为未实现候选；完整循环仍有scheduler，Fast-WAM scheduler为低成本tensor更新。不追加GPU/训练，非tile。

用户提出mega Transformer融合验证的研究思路。昂贵target方向求值可批量化，但完整循环仍有scheduler；长单也没有净收益，不能只把双并发当根因。源码显示candidate每层history index_select/cat与host同步候选；已保存shape推导并明确非实测HBM/峰值/因果。Fast-WAM可概念迁移方向投机，其compiled原基线与scheduler也必须公平对齐。未实现/启动新实验。

下一步：建议仅CPU整理共享KV、同步和候选隔离数据流/最小对照合同，先单需求长历史、双需求作压力辅检，并核对Fast-WAM方向投机接口；新kernel实现、GPU拆解及训练另确认，保持success/环境步数/E2E目标。

[讨论与源码/原始来源](VERIFIER-SYSTEMS-FASTWAM-DUAL-DEMAND-20261011.md)；[CPU证据](../../../review/OMISSIONS.md)；[完成的GPU成本](FIXED-LOAD-COST-RESULT-20261011-213036.md)；Knowledge CHAT-20261011-SPECULATIVE-VERIFIER-SYSTEMS-FASTWAM-DUAL。

## 65. KV复制/元数据同步最小归因方案与batch澄清（2026-10-12T00:09:00+08:00）

投机采样仅CPU完成KV/同步归因方案：7节点×2CFG一次batch verifier非14次target前向；两路历史逐层gather/index_select物化14行再cat。四臂R/H/C/HC只在真实第0层固定QKV隔离元数据外提与重复展开，C非最终共享KV。长单主测、长双压力、短272子集控制；拟0episode/0EGL/0完整chunk、51target前向+3702micro调用、45分钟上限，GPU未授权且包装未实现。新增512token文本投影/KV重复候选线索，非根因结论；不训练/mega/tile。

用户追问已batch为何14份：一棵K2B2树一次批量target verifier，非7/14次串行前向；7个候选×2CFG的当前Q/K/V各行不同，同路历史却在每层index_select物化7遍，再cat为14行。共享存储不自动去掉各Q的数学工作，不能把候选拼成可跨行attention的一个序列。

本步只读检查源码和既有6660个candidate层事件、prompt形状及fixture元数据/大小；未加载1.5GB PT、未捕获新QKV、未实现新包装、无CUDA API/GPU/训练。四臂第0层R/H/C/HC保留同一原SDPA和cat，setup收费，C是14行展开缓存而非最终两路共享；真实长单主测，短历史272子集仅合成长度控制，长双辅检。51target前向/3702额外micro调用/45分钟是未批准提案，不继承原168chunk授权。

通过标准：未来CPU真实接口/默认关闭/计数/归属回归先过；GPU逐bit输出、dtype/stride/backend、有效cache/RNG和候选隔离；OFF四重复MAD≤7%/dual重叠≥90%；长单≥5%且四顺序同向只标局部pilot，负结果可有效，不证明成功率/环境步数/E2E。失败停止不补；文本投影/512token cross-attention K/V、FFN/GEMM、RoPE转换等仍是未归因候选。

下一步：先确认私有默认关闭包装及CPU回归实现范围，过关后再单独确认45分钟GPU启动；不实现mega Transformer、不训练，真正共享prefix或文本KV复用只按未来归因另规划。

[完整方案](VERIFIER-KV-SYNC-ATTRIBUTION-PLAN-20261012.md)；[机器合同](../../../review/OMISSIONS.md)；[仅CPU静态检查](../../../review/OMISSIONS.md)；Knowledge CHAT-20261012-SPECULATIVE-KV-SYNC-ATTRIBUTION-CPU-PLAN。

## 66. 过夜实现与自动GPU授权（2026-10-12T01:29:15+08:00）

投机采样KV/同步归因过夜实现与自动GPU授权：PASS_AUTHORIZATION_AND_IMPLEMENTATION_READY；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。控制器新鲜CPU回归→稳定空闲资源→一次GPU归因→清理与晨间简报。失败停止不补跑，08:30结束供用户验收。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：控制器新鲜CPU回归→稳定空闲资源→一次GPU归因→清理与晨间简报。失败停止不补跑，08:30结束供用户验收。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-005214-AUTH。

## 67. 真实接口CPU回归通过（2026-10-12T01:31:24+08:00）

投机采样KV/同步归因真实接口CPU回归通过：PASS；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。CPU全通过；后台等待稳定空闲资源，最迟07:45启动一次45分钟GPU，失败停不补跑。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：CPU全通过；后台等待稳定空闲资源，最迟07:45启动一次45分钟GPU，失败停不补跑。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-005214-CPU。

## 68. CPU通过、后台资源等待（2026-10-12T01:31:24+08:00）

投机采样KV/同步归因CPU通过、后台资源等待：WAITING；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。稳定15个空闲样本后自动开始；到07:45仍无窗口则0GPU结束，晨间简报说明。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：稳定15个空闲样本后自动开始；到07:45仍无窗口则0GPU结束，晨间简报说明。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-005214-WAIT。

## 69. 关卡失败停止（2026-10-12T01:40:56+08:00）

投机采样KV/同步归因关卡失败停止：FAIL；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。先CPU定位失败或检查已通过的OFF结果；失败停止不补跑，修复后新GPU预算另确认；不自动共享kernel/mega或训练。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：先CPU定位失败或检查已通过的OFF结果；失败停止不补跑，修复后新GPU预算另确认；不自动共享kernel/mega或训练。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-005214/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-005214-FAILED。

## 70. 过夜实现与自动GPU授权（2026-10-12T01:47:01+08:00）

投机采样KV/同步归因过夜实现与自动GPU授权：PASS；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。正式控制器新鲜59接口+26归属CPU回归（含live NVML v2）通过，再等待15稳定空闲样本，执行原授权唯一一次45分钟GPU；实际GPU任一关失败停止不重试。0新episode/0EGL/0完整chunk，无训练/mega。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：正式控制器新鲜59接口+26归属CPU回归（含live NVML v2）通过，再等待15稳定空闲样本，执行原授权唯一一次45分钟GPU；实际GPU任一关失败停止不重试。0新episode/0EGL/0完整chunk，无训练/mega。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-AUTH。

## 71. 真实接口CPU回归通过（2026-10-12T01:47:17+08:00）

投机采样KV/同步归因真实接口CPU回归通过：PASS；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。CPU全通过；后台等待稳定空闲资源，最迟07:45启动一次45分钟GPU，失败停不补跑。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：CPU全通过；后台等待稳定空闲资源，最迟07:45启动一次45分钟GPU，失败停不补跑。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-CPU。

## 72. CPU通过、后台资源等待（2026-10-12T01:47:18+08:00）

投机采样KV/同步归因CPU通过、后台资源等待：WAITING；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。稳定15个空闲样本后自动开始；到07:45仍无窗口则0GPU结束，晨间简报说明。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：稳定15个空闲样本后自动开始；到07:45仍无窗口则0GPU结束，晨间简报说明。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-WAIT。

## 73. 资源门通过、开始GPU（2026-10-12T01:47:34+08:00）

投机采样KV/同步归因资源门通过、开始GPU：PASS；GPU启动=false，target预留0/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。启动两个干净模型worker，45分钟计时；无EGL，只恢复既有长fixture并有界捕获。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：启动两个干净模型worker，45分钟计时；无EGL，只恢复既有长fixture并有界捕获。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-START。

## 74. 原快照恢复与首次batch verifier捕获通过（2026-10-12T01:49:00+08:00）

投机采样KV/同步归因原快照恢复与首次batch verifier捕获通过：PASS；GPU启动=true，target预留46/51、micro预留0/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。进入逐bit/候选隔离关；不开始性能对照直到通过。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：进入逐bit/候选隔离关；不开始性能对照直到通过。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-CAPTURE。

## 75. 四臂逐bit与候选隔离通过（2026-10-12T01:49:09+08:00）

投机采样KV/同步归因四臂逐bit与候选隔离通过：PASS；GPU启动=true，target预留46/51、micro预留48/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。进入各条件warmup与四平衡顺序OFF主测；setup收费，负结果不算错误。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：进入各条件warmup与四平衡顺序OFF主测；setup收费，负结果不算错误。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-EQUIVALENCE。

## 76. 关卡失败停止（2026-10-12T01:50:09+08:00）

投机采样KV/同步归因关卡失败停止：FAIL；GPU启动=true，target预留48/51、micro预留2890/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。先CPU定位失败或检查已通过的OFF结果；失败停止不补跑，修复后新GPU预算另确认；不自动共享kernel/mega或训练。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：先CPU定位失败或检查已通过的OFF结果；失败停止不补跑，修复后新GPU预算另确认；不自动共享kernel/mega或训练。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-FAILED。

## 77. 整轮停止后的CPU部分结果分析与包装审计（2026-10-12T02:01:42+08:00）

投机采样KV/同步归因整轮停止后的CPU部分结果分析与包装审计：FAIL；GPU启动=true，target预留48/51、micro预留2890/3702，0episode/0EGL/0完整chunk，45分钟硬限。长单主测/双需求压力，R/H/C/HC保留原batch/SDPA/cat，C非最终共享KV；不训练/mega、非tile。先CPU补任务包装的双worker结束屏障、逐路径job隔离及加载状态更新回归；保持90%/预算不变，不新GPU。再规划真正两路共享prefix/元数据失效键及完整verifier收益核验，GPU或训练须另确认。

CPU真实接口/冻结检查→稳定空闲GPU→恢复/逐bit/隔离→干净OFF→独立ON→清理与简报。Torch profiler只在OFF完成后启用，不把模型从nsys包装器启动；无额外模型/调用，无训练。本job的过夜授权不改写原CPU-only proposal历史，也不扩大为6–8小时GPU计算。

下一步：先CPU补任务包装的双worker结束屏障、逐路径job隔离及加载状态更新回归；保持90%/预算不变，不新GPU。再规划真正两路共享prefix/元数据失效键及完整verifier收益核验，GPU或训练须另确认。

[本关](../../../review/OMISSIONS.md)；[简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；[合同](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-PARTIAL。

### 本关部分GPU数值与下一目标（2026-10-12T02:03:33+08:00；不改变FAIL终态）

第0层29次固定真实QKV（含setup，每臂4平衡重复）：长单R/H/C/HC中位86.900/83.525/43.726/39.781ms；C -49.7%、HC -54.2%，配对4块同向。长双151.159/161.550/83.002/75.613ms；C -45.1%、HC -50.0%，H +6.9%。仅局部pilot，不乘30、不推环境步数/成功率或E2E。C仍14行历史展开缓存，不是最终两路共享KV；双/单C/HC仍约1.90，压力未消失。

完整条件：长单、长双、272合成短单，各臂MAD≤7%；短双H第二块窗口重叠84.14%<90%，整轮OFF未通过，ON未跑。按失败关停止、不补GPU；真实接口59+26项回归、恢复/逐bit/候选隔离已通过，GPU48target/2890micro、155.43秒，0episode/0EGL/0完整chunk，无训练/mega、非tile，清理PASS。

下一步只建议CPU补双worker结束屏障、显式job路径隔离（归属测试脚本相同但旧CPU-copy拓扑metadata被改写）、spawn后状态即时更新回归；不新GPU。之后设计两路共享prefix/失效键与完整verifier实测，避免全30层14行常驻10.305GiB/worker；GPU和训练另确认。最终冻结target闭环成功率/环境步数非劣及E2E更快仍未证明。

[HW详细结果与包装审计](../../../review/OMISSIONS.md)；[晨间简报](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/MORNING_BRIEF.md)；Knowledge CHAT-P0-KV-SYNC-ATTRIBUTION-20261012-014447-PARTIAL。

## 78. 用户纠正：部署单需求，双EGL仅为采集吞吐（2026-10-12T05:43:51+08:00）

用户明确：投机采样实际部署/速度验收只看单需求，双独立EGL仅为LIBERO采集吞吐，不是部署性能目标。取消双GPU-worker并发压力、90%重叠及结束屏障作为后续主线验收/优化要求；模型内部7节点×2CFG的一次batch14验证保留。旧0EGL双需求压力实验FAIL原样保留，长单C/HC局部pilot不等于单模型独占部署/完整模型收益。先CPU制定单model-worker、无peer的C/HC历史KV复用与失效键方案，重估单模型显存；路径隔离/状态更新只作必要可靠性修复。取消双需求同步/结束屏障优化优先级，不新GPU、不训练、不mega。随后单独确认完整verifier单需求验证，再以冻结target闭环成功率、环境步数、E2E验收。

旧双需求是0EGL的两推理worker人工压力，不是两路EGL；不追改旧FAIL或90%门。后续无peer单模型下单独核验C/HC与完整verifier，旧长单两模型驻留的局部收益不能直接替代该验收。

下一步：先CPU制定单model-worker、无peer的C/HC历史KV复用与失效键方案，重估单模型显存；路径隔离/状态更新只作必要可靠性修复。取消双需求同步/结束屏障优化优先级，不新GPU、不训练、不mega。随后单独确认完整verifier单需求验证，再以冻结target闭环成功率、环境步数、E2E验收。

[范围纠正与缓存设计计划](DEPLOYMENT-SINGLE-DEMAND-SCOPE-20261012.md)；Knowledge CHAT-20261012-SPECULATIVE-SINGLE-DEMAND-SCOPE。

90%门澄清（2026-10-12T05:45:30+08:00）：它是人工双推理worker时间窗交集/较长窗口的并发有效性门，不是投机接受率；昨晚24.086/28.625ms=84.14%。单需求部署不需要该门，原FAIL保留。详见本节范围报告新增定义。

## 79. 单worker C/HC复用、失效与完整verifier预算：CPU方案（2026-10-12T06:09:36+08:00）

仅CPU完成投机采样单worker C/HC复用、失效/显存与完整verifier方案，32项静态核查通过；不等于真实接口回归/GPU通过。缓存仍为14行展开，30层新增payload10.305GiB；写入/rollback/eviction/reset/观测替换/row map或模型代次变化前失效，不按长度猜命中。新GPU提案仅1模型无peer、R/C/HC全30层真实变化29输入、三臂各6重复、0episode/EGL/full policy chunk，895已分配/900前向硬上限、45分钟，未授权未启动。旧GPU FAIL保留，初稿预算算术失败更正留档；不训练/mega/公共模型修改，非tile。

C/HC定义：每段只读history一次gather/index_select缓存14行；HC额外外提metadata/行检查，当前Q/K/V、cat和原attention均保留。普通update=0仍会写/rollback，不能按长度猜复用。逐层全模型适配器及真实29树capture尚未实现；不将旧单层pilot写成全模型收益。

下一步：先确认仅CPU实现任务专用、默认关闭的单worker全层C/HC adapter、只读租约/失效、全29真实变化query捕获与监督/真实接口回归；公共模型与原batch/attention不改。CPU回归通过后，再单独确认≤900 target、45分钟、1 worker、0episode/EGL/full chunk的GPU验证；当前不启动GPU或训练。

[详细方案与源码解释](SINGLE-WORKER-CHC-KV-PLAN-20261012.md)；[CPU静态核查](../../../review/OMISSIONS.md)；Knowledge CHAT-20261012-SPECULATIVE-SINGLE-WORKER-CHC-CPU-PLAN。

## 80. 默认关闭的单worker全层C/HC：实现与真实接口CPU回归（2026-10-12T08:24:55+08:00）

投机采样仅CPU实现默认关闭的单worker全30层C/HC实例包装、只读租约/全层失效与变化query捕获；81项真实接口回归通过。小尺寸CPU公开模型29变化查询C/HC与R逐bit、13行隔离、有效KV/metadata/RNG不变，非训练target GPU证明；192冻结文件不改，CPU失败留档。0新GPU/排队/训练，公共模型/batch14/原SDPA不变，非tile。下一步单独确认1 worker、895/900target、45分钟、0episode/EGL/full chunk，再补私有监督器/新job CPU预检后验证完整verifier收益与显存。

- C每层只gather/展开一次，仍动态metadata/row_check；HC外提该控制工作。原当前QKV/RoPE/cat/SDPA/后续模型计算不变，非真正共享prefix kernel。
- 按公开写路径全层撤销租约；同地址restore、update0 rollback/eviction也失效。固定CFG rows保持原值；首candidate历史在RAM保存，拒绝29query期间的历史变化。
- 无version inference池/参数、未声明.data/裸指针写入口不能安全复用；GPU真实历史可追踪性仍需核验。
- CPU001/003夹具失败与002/004通过全部保留；不将旧micro收益乘30，不宣称GPU/E2E/环境步数通过。

下一步：单独确认GPU预算（1 worker无peer，895已分配/900 target前向硬上限，45分钟安全限时，0episode/EGL/full policy chunk）；再新建job并补齐私有监督器、真实子进程CPU预检及资源/归属/限时检查，通过后按恢复→数值/失效→全verifier成本推进；失败停、不补跑、不训练。

[完整结果/边界](SINGLE-WORKER-CHC-CPU-RESULT-20261012.md)；[81项CPU结果](../../../review/OMISSIONS.md)；[实现说明](../../../投机采样/experiments/single_worker_chc/README.md)；Knowledge CHAT-20261012-SPECULATIVE-SINGLE-WORKER-CHC-CPU-IMPLEMENTATION。

## 81. 单worker预算授权与任务监督器实现（2026-10-12T08:50:17+08:00）

投机采样单worker预算授权与任务监督器实现：PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU，GPU启动=false，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。补齐监督器已实现；新job先做进程归属/限时/预算/真实接口及同worker入口CPU预检，通过后按本轮预算启动一次GPU，不排无限队列、不训练。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：补齐监督器已实现；新job先做进程归属/限时/预算/真实接口及同worker入口CPU预检，通过后按本轮预算启动一次GPU，不排无限队列、不训练。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-20261012-085017-AUTH。

## 82. 新job真实CPU预检通过（2026-10-12T08:52:58+08:00）

投机采样新job真实CPU预检通过：PASS，GPU启动=false，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-20261012-085017-CPU。

## 83. 空闲资源关通过、启动一次GPU（2026-10-12T08:55:49+08:00）

投机采样空闲资源关通过、启动一次GPU：PASS，GPU启动=true，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-20261012-085017-START。

## 84. 关卡失败停止（2026-10-12T08:57:05+08:00）

投机采样关卡失败停止：FAIL，GPU启动=true，target预留/尝试/完成=52/52/52，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-20261012-085017-FAILED。

## 85. 单worker全层C/HC GPU失败与CPU导入路径复盘（2026-10-12T09:06:11+08:00）

投机采样单worker全层C/HC新job：CPU 32+81项及30层worker probe通过，但真实GPU因私有包装只认可wan_va.modules.model、误拒服务器modules.model别名而FAIL。GPU加载至清理75.812秒，target尝试/完成52/52，0episode/EGL/完整chunk/训练；等价性/失效/隔离/成本均未运行，无新加速结论。CPU58项诊断复现同错误且仅模块名条件失败；197旧文件+15私有实现冻结不变，本任务GPU已清理，不补跑，非TILE。下一步先确认仅CPU修复私有包装的源码校验导入别名、补真实loader路径的正/负回归，并把接口准入移到52次捕获之前。公共源码/batch14/原SDPA/冻结权重不改；CPU通过后另确认新job GPU预算，本次不修复补跑、不训练。

CPU测试覆盖缺口归因到私有模块名准入；不改已执行源码和原始FAIL，不将diagnosis PASS混写成GPU PASS。

下一步先确认仅CPU修复私有包装的源码校验导入别名、补真实loader路径的正/负回归，并把接口准入移到52次捕获之前。公共源码/batch14/原SDPA/冻结权重不改；CPU通过后另确认新job GPU预算，本次不修复补跑、不训练。

[结果与复盘](SINGLE-WORKER-CHC-GPU-RESULT-20261012.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-20261012-085017-POSTMORTEM-CPU。

## 86. 生产loader合法路径与capture前准入仅CPU修复（2026-10-12T09:33:45+08:00）

投机采样单worker全层C/HC合法生产导入路径仅CPU修复完成：独立私有修订严格接受modules.model/wan_va.modules.model，仍核对固定源码/真实导出class/原forward可执行结构和defaults/原SDPA/eval/cache及全层。接口准入提前到Runtime计数/ready与capture读取fixture/reset/infer之前，不安装优化、不跑target；坏接口0前向拒绝。真实未mock生产loader37项正负回归+原81项回归+同worker生产loader全30层29变化输入R/C/HC CPU probe全部通过，总113.305秒；随机小checkpoint非训练target GPU证明。269份旧源码/失败证据及公共模型/batch/attention不变；0新GPU/排队/训练，非TILE。下一步单独确认一次全层C/HC真实GPU验证预算：建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练。确认后才补新修订的默认关闭GPU入口及新job预检，不复用旧FAIL job或旧CPU gate；接口前置准入→恢复/捕获→数值/失效/隔离→六块干净OFF→独立ON→清理，失败停不补跑。

旧GPU误拒的根因已修复，118项回归及真实生产loader同worker CPU probe通过；不继承GPU授权，不证明真实target GPU数值/显存/性能。

下一步单独确认一次全层C/HC真实GPU验证预算：建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练。确认后才补新修订的默认关闭GPU入口及新job预检，不复用旧FAIL job或旧CPU gate；接口前置准入→恢复/捕获→数值/失效/隔离→六块干净OFF→独立ON→清理，失败停不补跑。

[CPU结果](SINGLE-WORKER-CHC-LOADER-FIX-CPU-RESULT-20261012.md)；Knowledge CHAT-20261012-SPECULATIVE-SINGLE-WORKER-CHC-LOADER-FIX-CPU。

## 87. 新版本单worker预算授权与GPU入口实现（2026-10-12T10:27:20+08:00）

投机采样新版本单worker预算授权与GPU入口实现：PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU，GPU启动=false，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新版本GPU入口已补齐；新job先过进程/预算/限时、真实生产loader正负例及同worker CPU预检，再按本次授权每3分钟等待独占空闲，自动启动一次GPU。失败停、不补跑、不训练。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新版本GPU入口已补齐；新job先过进程/预算/限时、真实生产loader正负例及同worker CPU预检，再按本次授权每3分钟等待独占空闲，自动启动一次GPU。失败停、不补跑、不训练。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-AUTH。

## 88. 新版本新job生产loader/监督器CPU预检通过（2026-10-12T10:30:44+08:00）

投机采样新版本新job生产loader/监督器CPU预检通过：PASS，GPU启动=false，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-CPU。

## 89. 空闲资源关通过、启动一次GPU（2026-10-12T10:33:24+08:00）

投机采样空闲资源关通过、启动一次GPU：PASS，GPU启动=true，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-START。

## 90. 真实GPU生产接口前置准入通过（2026-10-12T10:33:51+08:00）

投机采样真实GPU生产接口前置准入通过：PASS，GPU启动=true，target预留/尝试/完成=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-ADMISSION。

## 91. 关卡失败停止（2026-10-12T10:34:21+08:00）

投机采样关卡失败停止：FAIL，GPU启动=true，target预留/尝试/完成=52/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-FAILED。

## 92. 新版本GPU真实准入通过、独占中断与账本澄清（2026-10-12T10:42:10+08:00）

投机采样单worker全层C/HC新版本GPU：159项CPU回归及真实生产loader同worker probe通过，真实target GPU生产modules.model全30层前置准入通过，旧导入别名误拒未再出现。稳定空闲启动后，hlj的PID1009191进入GPU，独占关失效，约57.032秒停止并清理本任务，未信号对方。capture中断非PASS，C/HC数值/失效/隔离/显存/成本与ON未运行，无新加速结论。父账本0是未收到阶段确认的旧值，不是实际0前向；实际数学已开始，精确次数未落盘未知，本关源码硬限52。298前置文件+17已执行私有文件及49份raw job不改，0episode/EGL/完整chunk/训练，非TILE。建议下一步先只CPU补中断时账本持久化及真实worker终止回归，同时协调可靠的45分钟独占GPU窗口。CPU通过后另确认新job：仍单worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停。当前FAIL不复用、不补跑；本轮没有自动下一实验。

这是外部资源进入导致中止，非模型数值或C/HC成本失败。raw父账本0不是最终实际0，精确次数未知，单capture源码cap52；原始记录保留，不改FAIL。

建议下一步先只CPU补中断时账本持久化及真实worker终止回归，同时协调可靠的45分钟独占GPU窗口。CPU通过后另确认新job：仍单worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停。当前FAIL不复用、不补跑；本轮没有自动下一实验。

[结果与澄清](SINGLE-WORKER-CHC-GPU-V2-RESULT-20261012.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V2-20261012-102720-POSTMORTEM-CPU。

## 93. 中断写前账本与真实worker终止仅CPU修复通过（2026-10-12T11:24:32+08:00）

投机采样单worker中断计数持久化仅CPU修复完成：默认关闭独立修订，before模型调用持久化预算意图、返回后持久化host-return确认，序号/hash-chain/checksum/job-plan/PID-start身份绑定，每事件fsync，失败在公共数学前止步；TERM仅设标志、安全边界写退出，KILL恢复最多一个未确认意图，缺失/坏记录不套父旧0、不退款/续跑。162项回归(44新增含34日志+10真实worker/数值、37生产loader、81既有)及新同worker FP32 probe通过；全30层29变化query R/C/HC FP32/BF16逐bit一致、history/RNG不变。9个实际CPU serve退出场景通过，KILL中第2次可恢复2/1/1、29次结束未ack可恢复29/29/0；不冒充CUDA完成。CPU首次292.089秒，371旧文件+15私有执行文件/146旧HW记录不变，旧GPU次数仍未知不回填。0新GPU/排队/episode/EGL/完整chunk/训练，非TILE。下一步先单独确认独占方式/窗口和新GPU预算。建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；确认后才将日志与中断恢复接入新默认关闭GPU入口/监督器，过新job CPU预检，再恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停不补跑。当前不自动排队/GPU/训练/mega，也不复用旧FAIL。

旧GPU失去的精确次数仍未知；本修订不回填或改FAIL。记录的是预算意图/host return，KILL最后一项最多未确认，不冒充CUDA完成。fsync计入原序列timer，公共数学/attention不变。

下一步先单独确认独占方式/窗口和新GPU预算。建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；确认后才将日志与中断恢复接入新默认关闭GPU入口/监督器，过新job CPU预检，再恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停不补跑。当前不自动排队/GPU/训练/mega，也不复用旧FAIL。

[结果](SINGLE-WORKER-CHC-COUNTER-PERSISTENCE-CPU-RESULT-20261012.md)；Knowledge CHAT-20261012-SPECULATIVE-SINGLE-WORKER-CHC-COUNTER-PERSISTENCE-CPU。

## 94. 新版本单worker预算授权与GPU入口实现（2026-10-12T14:26:32+08:00）

投机采样新版本单worker预算授权与GPU入口实现：PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新v3入口已接入持久化计数；先新job CPU回归及真实worker终止/FP32-BF16验证，再检查空闲后启动一次单worker GPU。共享模式不改，忙时仅启动前每3分钟等待；启动后他人进入即停，不补跑、不训练。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新v3入口已接入持久化计数；先新job CPU回归及真实worker终止/FP32-BF16验证，再检查空闲后启动一次单worker GPU。共享模式不改，忙时仅启动前每3分钟等待；启动后他人进入即停，不补跑、不训练。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V3-20261012-142632-AUTH。

## 95. 关卡失败停止（2026-10-12T14:31:06+08:00）

投机采样关卡失败停止：FAIL，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V3-20261012-142632-FAILED。

## 96. 单worker C/HC v3 CPU失败复盘：GPU未启动（2026-10-12T14:36:10+08:00）

投机采样新C/HC v3 CPU预检FAIL，0GPU/排队/训练：49监督器+44日志/真实worker通过，loader35/37，2负例因新版本flag清理/私有路径报错契约未同步而断言失败；旧job实际拒绝，无数值失败证明。130项128通过，245秒后清理；81及独立FP32 probe/GPU全部未运行。657冻结文件+21源码/旧FAIL不变；停止不补跑，非TILE。

两个旧job负例均正确拒绝，但新版本环境flag清理/报错契约断言未同步，是本次包装工程遗漏；不放宽入口、不推断数学不正确或GPU通过。原始FAIL保留、真实GPU测试未运行。

下一步：本job已停止，不补跑/复用/排队。下一步建议单独授权仅CPU同步这两个负例的版本flag清理和私有路径报错契约，补真实CLI/旧job拒绝回归；不放宽门、不改公共模型/batch/attention。CPU修复通过后，再单独确认新GPU job及独占窗口/原900次45分钟预算。

[报告](SINGLE-WORKER-CHC-GPU-V3-RESULT-20261012.md)；[只读审计](../../../review/OMISSIONS.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V3-20261012-142632-POSTMORTEM-CPU。

## 97. 新版本单worker预算授权与GPU入口实现（2026-10-12T14:58:20+08:00）

投机采样新版本单worker预算授权与GPU入口实现：PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。按用户持续修复及并行讲解授权：只修私有负例flag隔离/路径断言，增加真实CLI导入前拒绝证明；新job CPU关通过后自动检查空闲并启动一次GPU，累计仍≤900前向/45分钟。共享模式不改，不训练/新prefix kernel/公共模型改动；旧FAIL不重写。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：按用户持续修复及并行讲解授权：只修私有负例flag隔离/路径断言，增加真实CLI导入前拒绝证明；新job CPU关通过后自动检查空闲并启动一次GPU，累计仍≤900前向/45分钟。共享模式不改，不训练/新prefix kernel/公共模型改动；旧FAIL不重写。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-AUTH。

## 98. 新版本新job生产loader/监督器CPU预检通过（2026-10-12T15:05:08+08:00）

投机采样新版本新job生产loader/监督器CPU预检通过：PASS，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-CPU。

## 99. 按授权每3分钟等候GPU独占空闲（2026-10-12T15:05:09+08:00）

投机采样按授权每3分钟等候GPU独占空闲：WAITING_RESOURCE，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。按用户授权每3分钟只读检查，空闲后稳定独占确认再自动启动一次GPU；0本任务GPU，不信号外部PID。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：按用户授权每3分钟只读检查，空闲后稳定独占确认再自动启动一次GPU；0本任务GPU，不信号外部PID。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-WAIT。

## 100. 空闲资源关通过、启动一次GPU（2026-10-12T15:08:24+08:00）

投机采样空闲资源关通过、启动一次GPU：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/UNKNOWN/UNKNOWN，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-START。

## 101. 真实GPU生产接口前置准入通过（2026-10-12T15:08:55+08:00）

投机采样真实GPU生产接口前置准入通过：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-ADMISSION。

## 102. 关卡失败停止（2026-10-12T15:09:00+08:00）

投机采样关卡失败停止：FAIL，GPU启动=true，target预留/预算意图/已确认host返回=52/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-FAILED。

## 103. C/HC v4 CPU修复通过，真实GPU资源冲突停止（2026-10-12T15:16:38+08:00）

投机采样单worker C/HC v4：两处私有负例包装bug修复，215/215 CPU回归及同worker生产loader probe通过；只读subagent核验普通B2/verify B14、history B2/current B14，形状与共享思路已记录。GPU稳定空闲启动后实际target全30层生产接口准入PASS，随后非本任务PID1665051进入，监督器约35.995秒安全停止。持久化日志证明GPU前向意图/host返回/未确认=0/0/0，capture仅预留52；C/HC数值、显存、OFF/ON均未运行，无新性能结果。829冻结文件+21源码/旧FAIL及本FAIL保留，已清理无队列，0episode/EGL/完整chunk/训练，非TILE。

本轮独立只读subagent确认prefix两路CFG、当前十四行候选；C/HC仍缓存展开history，不是直接共享prefix kernel，prompt共享未做。[形状与下一步接口](SINGLE-WORKER-VERIFY-SHAPES-20261012.md)。CPU工程修复PASS，GPU资源冲突FAIL分别记录，不把target接口通过/可信0计数冒充C/HC数值或性能通过。

下一步：私有bug已经修复并通过CPU；当前需要协调连续独占GPU窗口，不继续修改公共模型/attention或反复在短暂空档抢跑。用户已授权持续修复，不再为每个CPU小修确认；后续新job不得复用本FAIL或弱化资源监控，并须把本轮已耗35.995秒计入累计≤900次/45分钟GPU预算。窗口明确后先核对剩余预算/新job预检，再恢复→数值/失效/隔离→六块OFF→独立ON。未获速度结果前不宣称加速，不训练/mega或自动共享prefix改造。

[报告](SINGLE-WORKER-CHC-GPU-V4-RESULT-20261012.md)；[只读审计](../../../review/OMISSIONS.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-145820-POSTMORTEM-RESOURCE。

## 104. 新版本单worker预算授权与GPU入口实现（2026-10-12T15:24:04+08:00）

投机采样新版本单worker预算授权与GPU入口实现：PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。用户新授权空了赶快测：沿用215项通过的私有实现，不改模型/batch/attention；新job CPU关重新验证后自动空闲启动。旧GPU35.995秒及0次前向由已冻结日志证明，CPU-only外层监督器扣除已耗时间，累计≤900前向/45分钟。失败停，不训练，不复用旧FAIL。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：用户新授权空了赶快测：沿用215项通过的私有实现，不改模型/batch/attention；新job CPU关重新验证后自动空闲启动。旧GPU35.995秒及0次前向由已冻结日志证明，CPU-only外层监督器扣除已耗时间，累计≤900前向/45分钟。失败停，不训练，不复用旧FAIL。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-AUTH。

## 105. 新版本新job生产loader/监督器CPU预检通过（2026-10-12T15:29:06+08:00）

投机采样新版本新job生产loader/监督器CPU预检通过：PASS，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-CPU。

## 106. 按授权每3分钟等候GPU独占空闲（2026-10-12T15:29:07+08:00）

投机采样按授权每3分钟等候GPU独占空闲：WAITING_RESOURCE，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。按用户授权每3分钟只读检查，空闲后稳定独占确认再自动启动一次GPU；0本任务GPU，不信号外部PID。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：按用户授权每3分钟只读检查，空闲后稳定独占确认再自动启动一次GPU；0本任务GPU，不信号外部PID。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-WAIT。

## 107. 空闲资源关通过、启动一次GPU（2026-10-12T15:32:23+08:00）

投机采样空闲资源关通过、启动一次GPU：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/UNKNOWN/UNKNOWN，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-START。

## 108. 真实GPU生产接口前置准入通过（2026-10-12T15:32:46+08:00）

投机采样真实GPU生产接口前置准入通过：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/0/0，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-ADMISSION。

## 109. 真实恢复与29变化输入捕获通过（2026-10-12T15:33:49+08:00）

投机采样真实恢复与29变化输入捕获通过：PASS，GPU启动=true，target预留/预算意图/已确认host返回=53/53/53，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。进入C/HC全30层29变化输出、十二类失效与13行隔离关；未通过不测速。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：进入C/HC全30层29变化输出、十二类失效与13行隔离关；未通过不测速。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-CAPTURE。

## 110. 关卡失败停止（2026-10-12T15:40:14+08:00）

投机采样关卡失败停止：FAIL，GPU启动=true，target预留/预算意图/已确认host返回=210/183/179，895已分配/900硬限，45分钟（含加载至清理）。1 model无peer，0episode/EGL/完整chunk，R/C/HC原batch14/原SDPA/公共模型不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU关数值、失效、隔离/显存→六块干净OFF→独立ON→清理。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/plan.json)；[简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028/BRIEF.md)；Knowledge CHAT-P0-SINGLE-WORKER-CHC-V4-20261012-152028-FAILED。

## 111. CFG两路保留、七候选历史零拷贝共享S实现启动（2026-10-12T16:04:22+08:00）

用户确认只共享七候选、不合并CFG。新S私有默认关闭路径已实现：直接读batch2原池+valid_slots，current/Q保持14，不生成history14/fullKV14，联合softmax；CPU首轮24项通过，尚无新S GPU或性能结论。旧C/HC基线因外部PID进入停止：两臂29真实输出逐位一致、前十失效类通过，完整验证及OFF未完成。

本轮S采用batch2原池零拷贝和有效索引，避免batch14历史及fullKV14。CFG与当前候选保持独立；prompt不共享。[形状与下一步接口](SINGLE-WORKER-VERIFY-SHAPES-20261012.md)。旧C/HC部分GPU通过与外部资源中断分别保留，新S CPU数学/接口不是S GPU验收。

下一步：补齐新S job CPU入口/失效回归；单worker新GPU按剩余715次target及累计45分钟余额推进，先kernel/真实全层数值再测OFF；S严格逐位不过则停止或先获得仅诊断性能分支授权，不部署/训练。

[报告](SHARED-PREFIX-CFG2-PLAN-20261012.md)；[只读审计](../../../review/OMISSIONS.md)；Knowledge CHAT-SHARED-PREFIX-CFG2-IMPLEMENTATION-20261012。

## 112. 新版本单worker预算授权与GPU入口实现（2026-10-12T16:23:55+08:00）

投机采样新版本单worker预算授权与GPU入口实现：AUTHORIZED_CPU_PENDING，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v1-20261012-162355/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v1-20261012-162355/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V1-20261012-162355-AUTH。

## 113. 关卡失败停止（2026-10-12T16:25:28+08:00）

投机采样关卡失败停止：FAIL，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v1-20261012-162355/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v1-20261012-162355/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V1-20261012-162355-FAILED。

## 114. 新版本单worker预算授权与GPU入口实现（2026-10-12T16:28:05+08:00）

投机采样新版本单worker预算授权与GPU入口实现：AUTHORIZED_CPU_PENDING，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v2-20261012-162805/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v2-20261012-162805/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V2-20261012-162805-AUTH。

## 115. 新版本新job生产loader/监督器CPU预检通过（2026-10-12T16:31:06+08:00）

投机采样新版本新job生产loader/监督器CPU预检通过：PASS，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v2-20261012-162805/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v2-20261012-162805/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V2-20261012-162805-CPU。

## 116. 新版本单worker预算授权与GPU入口实现（2026-10-12T16:34:53+08:00）

投机采样新版本单worker预算授权与GPU入口实现：AUTHORIZED_CPU_PENDING，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-AUTH。

## 117. 新版本新job生产loader/监督器CPU预检通过（2026-10-12T16:37:50+08:00）

投机采样新版本新job生产loader/监督器CPU预检通过：PASS，GPU启动=false，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-CPU。

## 118. 空闲资源关通过、启动一次GPU（2026-10-12T16:41:23+08:00）

投机采样空闲资源关通过、启动一次GPU：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/UNKNOWN/UNKNOWN，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-START。

## 119. 真实GPU生产接口前置准入通过（2026-10-12T16:41:56+08:00）

投机采样真实GPU生产接口前置准入通过：PASS，GPU启动=true，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-ADMISSION。

## 120. 共享prefix kernel独立数值与隔离诊断（2026-10-12T16:42:06+08:00）

投机采样共享prefix kernel独立数值与隔离诊断：PASS_KERNEL_DIAGNOSTIC，GPU启动=true，target预留/预算意图/已确认host返回=0/0/0，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。kernel数学/隔离关通过（非target验收）；恢复真实fixture，验证29变化query全30层。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：kernel数学/隔离关通过（非target验收）；恢复真实fixture，验证29变化query全30层。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-KERNEL。

## 121. 真实恢复与29变化输入捕获通过（2026-10-12T16:43:19+08:00）

投机采样真实恢复与29变化输入捕获通过：PASS，GPU启动=true，target预留/预算意图/已确认host返回=53/53/53，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。进入冻结HC与S全30层数值、十二类失效及13行隔离；严格门不降低。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：进入冻结HC与S全30层数值、十二类失效及13行隔离；严格门不降低。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-CAPTURE。

## 122. 关卡失败停止（2026-10-12T16:43:56+08:00）

投机采样关卡失败停止：FAIL，GPU启动=true，target预留/预算意图/已确认host返回=111/111/111，585已分配/715本轮硬限，36分钟（含加载编译至清理）；保留历史183意图/507.291秒，累计仍≤900/45分钟。1 model无peer，0episode/EGL/完整chunk，R/HC原SDPA保持；S只替换私有候选历史attention，CFG2原池零拷贝、current14，公共模型源码不变；非TILE、未训练。失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

本轮只验证单worker全30层、29变化query的历史复用和完整verifier成本，取消双worker重叠/结束屏障作为主验收。写前日志不冒充CUDA完成，未知不套旧0；GPU kernel诊断→全30层数值/失效/隔离/显存→R/S六交错块干净OFF→清理；严格逐位不过且无诊断授权则停止。旧CPU未授权方案及旧FAIL保留，不把本轮授权写回旧记录。

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[本关](../../../review/OMISSIONS.md)；[合同](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)；[简报](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/BRIEF.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-FAILED。

## 123. 七候选共享KV真实GPU逐位失败、CPU误差归因与后续边界（2026-10-12T17:08:14+08:00）

投机采样CFG2/七候选共享S真实GPU已运行，但正式逐位门FAIL：29输出均有限、0/29逐位，full-output相对RMS0.395–4.520%（17/29>既有1%诊断上限）。同R QKV逐层attention局部差异0.0686–0.1059%，首层Q/currentK相同但attention先不同，支持数值路径变化+层间传播，不能证明无bug；29条是独立固定输入非S轨迹累积。CPU复算/只读子代理复核完成；GPU已清理，S最终状态/失效/隔离/OFF未到达，无新性能结论。111意图/111host返回、45直接attention、153.259秒，累计294/900及660.551/2700秒；不补跑、不训练、非TILE。

已实现CFG2原池零拷贝/current14，不造history14/fullKV14；不是七CTA片上复用。首层Q/currentK相同、attention先出现差异；同R QKV局部差异和后续层传播分开记录。29条是独立固定输入replay，不是S轨迹累积。真实S的最终状态/RNG、GPU失效/隔离及OFF未到达，不把kernel诊断当策略或性能通过。

本次GPU111意图/111host返回/0pending、45次独立attention意图、153.259秒；累计294/900、660.551/2700秒。GPU已清理，无自动队列/补跑，余额不是授权。CPU分析不启动GPU/模型/训练，公共模型和保护文件不改，非TILE。首轮CPU计数勘误：22而非24；最终v3为42+同worker58，失败版本和旧记录不覆盖。

下一步：建议先只CPU制定prefix-attention最小数值归因/修复方案，核查原SDPA backend与scale/exp2/归约/P转BF16路径，明确真实prefix/slots/currentV和状态对照。正式逐位门不降；新GPU/预算或数值非劣验收需另确认。先数值/不扰动通过，再六组R/S完整verifier OFF成本和显存；最终仍验收成功率、环境步数、E2E。当前无自动GPU、mega或训练。

[报告](SHARED-PREFIX-CFG2-NUMERICAL-RESULT-20261012.md)；[CPU复算](../../../review/OMISSIONS.md)；[只读独立复核](../../../review/OMISSIONS.md)；Knowledge CHAT-P0-SHARED-PREFIX-V3-20261012-163453-POSTMORTEM-NUMERICAL。

## 124. 共享KV数值归因与最小修复方案（仅CPU，未启动GPU）（2026-10-12T17:47:43+08:00）

投机采样共享KV数值方案仅CPU完成：原SDPA实际A100 backend与输出stride仍UNKNOWN；新kernel已FP32累加，已存PTX确认BF16概率/在线exp2/FMA。13项CPU代数/隔离检查、3种子概率精度×分块研究及只读复核支持优先排查概率舍入与分块交互，但不证明GPU修复。CFG2、七候选共享、batch14及原R正式字节门不变；无新GPU/队列/模型/训练，旧S仍FAIL且无正式性能结论。

确认差异：S已用FP32累加、BF16 MMA、P转BF16、BN64在线exp2/FMA，不能归咎于未证实的TF32。原SDPA实际backend和输出stride未观测；CPU单head固定logits研究与真实GPU根因/修复严格区分。先身份/数据，再layout-only（若有差异）、概率舍入、融合、指数、tile与精度；不同时改多项，不改参考或容差。

本步无新GPU/target/队列/模型/训练，公共模型和保护文件不改，非TILE。未来1worker/52target/最多24直接SDPA/15分钟仅为未授权提案；fixture三层166.5MiB、拟上限192MiB。累计仍294/900、660.551/2700秒；独立采集52后只余554＜原完整585，须未来显式整合capture或另确认预算，不偷删测试。旧S正式FAIL和原始失败保留；未获得正式速度/显存结论。

下一步：建议下一步只CPU实现任务专用默认关闭的原SDPA backend/输出布局记录器与真实QKV/prefix导出包装，补生产loader、默认关闭不扰动、恢复/计数/大小与独立存储回归；数值kernel暂不改，CPU中的GPU backend保留UNKNOWN。通过后单独确认最小GPU观测预算，再按真实输入选择一个候选修复；不自动GPU或训练。

[报告](SHARED-PREFIX-NUMERICAL-PLAN-CPU-20261012.md)；[CPU证据](../../../review/OMISSIONS.md)；[静态对照](../../../review/OMISSIONS.md)；[方案只读复核](../../../review/OMISSIONS.md)；Knowledge CHAT-SHARED-PREFIX-NUMERICAL-PLAN-CPU-20261012。

## 125. 共享KV真实backend/存储对照：诊断完成，S逐位门仍FAIL（2026-10-12T18:37:58+08:00）

共享KV数值差异已定位到attention计算路径变化：真实默认R为PyTorch Flash Attention；第0/8/29层、同一真实QKV下，S读取原CFG2共享池与复制CFG2历史逐位一致，相对R的单层RMS差异为0.06864%–0.10589%。原29输出哈希不变，数据、输出布局、局部cache版本/flags/RNG检查通过。S正式逐位门仍FAIL，尚无数值修复或性能验收；FP64抽样不支持把参考差异直接解释为数学精度下降。

本轮真实backend为Flash，R/S输出stride均[49152,3072,128,1]。同S kernel共享原池与复制CFG2历史三层2,064,384元素逐位相同；全部14行prefix值和原29输出hash保持一致。标准Efficient/MATH/cuDNN也不等于Flash，18切片FP64抽样中S没有比R更差；具体唯一主导舍入点仍未定。

v1早包装在0target/0direct时拒绝，34.033726秒计入预算且原失败/source保存；v2生产all30 CPU10项和GPU52target/24direct完成并清理，两次共112.700288秒。累计346/900、773.250910/2700秒，余554＜旧完整585计划；0episode/EGL/chunk/训练，不外推性能或任务质量。1589冻结文件与6实现文件复核通过，保护状态不改，非TILE。

下一步：若继续追求逐位一致，以实际PyTorch Flash Attention计算实现为基准，尽量只改K/V访存地址映射为CFG2共享历史+各候选当前K/V，保持原归约、softmax与舍入顺序；先小规模真实输入门，再覆盖全部30层/29query、state/RNG、12类失效、13行隔离，最后测完整verifier速度。本轮未定位唯一主导舍入点；不通过改参考backend或放宽容差宣称修复。原累计余554target，小于旧完整585计划，后续需重新安排验证预算并保留全部门。

[完整报告](SHARED-PREFIX-BACKEND-RESULT-20261012.md)；[Debug报告](../../debug/Debug-SHARED-PREFIX-BACKEND-20261012.md)；[GPU原始结果](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/FINAL_RESULT.json)；[CPU审计](../../../review/OMISSIONS.md)；Knowledge DEBUG-SHARED-PREFIX-BACKEND-20261012。

## 126. 共享KV直接原torch逐位对齐：全30层/29输出及失效隔离通过（2026-10-12T20:01:24+08:00）

共享KV已经直接对齐原始torch SDPA：复用当前PyTorch2.9对应FlashAttention计算、仅改六处KV读取地址，并移除实验编译中的fast-math。真实3层2,064,384元素、完整30层首个候选Q/K/V与attention输出、全部29个最终候选输出均逐位一致；12类缓存失效、R/HC/S的其他13行隔离、历史/metadata/RNG和原29输出hash检查通过。新S数值门PASS，性能与真实任务闭环尚未验收。

原Flash倒序M128/N64、线程内分母累加后最终归约、UNFUSE_FMA保持；6处KV取数替换，算术/同步源码字节未改。第一版fast-math剩1元素且dense/共享相同，第二版仅移除选项后3层全2,064,384元素与原torch逐位一致；反汇编确认原torch及新版本非FTZ/精确倒数校正，不声称同SASS或唯一一条指令因果。

CPU16项；GPU新target161意图、48直接attention、483.698125秒，闭合计数/清理通过。累计507/900、1256.949035/2700秒；0episode/EGL/chunk/训练。旧失败保留，原29hash/公共来源/TILE与保护状态不改。

S数值门PASS；六组OFF性能门/真实task闭环尚未执行，整体性能验收未通过。下一步：在相同原torch参考及逐位门下，制定下一轮有界完整verifier性能方案，比较R/S的时间、额外显存和稳定性；采用原先六组交叉顺序，不因数值通过就宣称加速。当前完整性能计划尚未执行，下一计划须携带累计预算，不能把旧585-forward计划直接重跑或默默删门。性能有收益后再考虑task0固定条件的闭环。

[完整报告](SHARED-PREFIX-TORCH-ALIGNMENT-RESULT-20261012.md)；[Debug报告](../../debug/Debug-SHARED-PREFIX-TORCH-ALIGNMENT-20261012.md)；[GPU结果](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/FINAL_RESULT.json)；[CPU审计](../../../review/OMISSIONS.md)；Knowledge DEBUG-SHARED-PREFIX-TORCH-ALIGNMENT-20261012。

## 127. 共享KV完整verifier六组性能对照完成（2026-10-12T20:54:43+08:00）

共享KV完整verifier六组R/S交叉对照完成：原torch中位5.170557秒、共享版3.359750秒，S/R=0.649785，耗时变化-35.02%；稳定正向 pilot。额外显存峰值891.035→257.085MiB。全部348个计时输出与原torch逐位一致，history/metadata/RNG及资源门通过；尚未验证任务成功率或E2E。监督器在已完成后的退出阶段FAIL，原失败保留，测量完成已独立CPU复核；收尾修复10项CPU检查通过，尚未GPU集成。

本轮保留完整R/S各29query预热、六组交叉顺序和原≤7%MAD/逐位门，所有348计时输出字节一致；旧12失效/隔离和原CUDA/source/binary冻结复用，未删门。

性能CPU9项、收尾修复CPU10项与独立bytes/计数/SHA/清理复查通过；父监督器完成后收尾FAIL原样保留，CPU核验仅报告已完整完成测量，不重跑GPU或改失败job。用户明确累计target从900扩至1000，旧计划不覆盖。实际461前向/379.375312GPU秒，累计968/1000及1636.324347/2700秒；0episode/EGL/完整chunk/训练。

下一步：先将已通过10项CPU检查的退出身份修复接入新私有监督包装，不复用或重跑失败job。先设计task0固定条件的小规模闭环，用冻结T/A_off及共享版本比较成功率、接受率、环境步数和E2E。当前回执只覆盖此次verifier测试，0episode上限与累计2700秒保持；闭环须制定新的明确预算后执行。

[完整报告](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)；[六组原始统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/OFF-RESULT.json)；[CPU审计](../../../review/OMISSIONS.md)；Knowledge CHAT-SHARED-PREFIX-TORCH-PERFORMANCE-20261012。

## 128. 共享KV短中档补测：短档混合、中档六组更快（2026-10-12T21:59:02+08:00）

短KV288/中KV1152原Torch六组完整verifier补测完成：短档中位3.114379→2.748547秒（-11.75%），但第1组慢5.56%，信号flat_or_mixed；中档3.834371→3.290483秒（-14.18%），六组均快，positive_pilot。两档660个实际计时输出/4,435,200元素与同档原Torch逐字节一致。新监督器正常PASS并清理，旧长档退出FAIL保留；累计1865/2200次、2007.138085/2700秒。仅验证verifier，prompt投影/cross-attention未新增共享，任务闭环/E2E未验证。

两档R/S各全部自然query预热、六组交叉顺序、原Torch逐位门和≤7%MAD门均保留；没有删除回退样本。短档按原方案追加2次profile，仅有限机制归因，首组OFF回退唯一原因未确定。中档18次真实cache回放，未合成或裁剪历史。

19项CPU预检与独立bytes/统计/journal/SHA/清理复查通过。退出身份修复仅在新私有监督器接入，新父监督器PASS；旧长档退出FAIL及源码原样保留。用户明确累计target1000→2200，GPU累计仍2700秒；新增897次/370.813737秒，累计1865/2200及2007.138085/2700秒。0episode/EGL/完整chunk/环境步/训练，公共源和受保护状态不改。

下一步：短KV仍不能保证稳定加速；优先针对短档的slot/row校验、索引生命周期和KV访存成本设计单独的CPU方案，保持原Torch逐位门，再制定有界验证。当前不自动改kernel、加测、实现prompt共享或启动任务闭环。任务成功率/接受率/E2E仍待有独立方案与预算的真实任务验证。

[完整报告](SHARED-PREFIX-SHORT-MIDDLE-RESULT-20261012.md)；[短档统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/short-OFF-RESULT.json)；[中档统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/middle-OFF-RESULT.json)；[CPU审计](../../../review/OMISSIONS.md)；Knowledge CHAT-SHARED-PREFIX-SHORT-MIDDLE-20261012。

## 129. 短KV行映射证明复用CPU完成，GPU三臂对照待准备（2026-10-12T22:49:30+08:00）

短KV行映射证明复用CPU原型完成：21项检查通过；实际CPU行检查scope的scalar次数30→1，原共享S0/优化S1的完整30层3个变化query、20,160元素逐字节一致。三档历史272/1136/2144的CPU接口、失效/投影期间突变/候选及CFG隔离检查通过。原history前后precheck、数学及CUDA v2不改。GPU速度/训练目标逐位门尚未验证；本轮0GPU，累计1865/2200次、2007.138085/2700秒不变。三臂624次/600秒GPU方案仅设计，执行包装和预算回执未完成。

先准备默认关闭的短档R/S0/S1三臂GPU执行包装并完成其CPU计数/计时/恢复/预算拒绝门；全部通过后，再以绑定新plan哈希的明确回执确认累计target2200→2500，GPU累计仍2700秒。随后执行同条件六组，保留全部query预热、原Torch逐位门和全部样本。当前不自动GPU、prompt共享或真实任务闭环。

[完整报告](SHARED-PREFIX-ROWPROOF-CPU-RESULT-20261012.md)；Knowledge CHAT-SHARED-PREFIX-ROWPROOF-CPU-20261012。

## 130. 短KV三臂包装CPU预检完成，GPU预算待确认（2026-10-12T23:18:11+08:00）

短KV三臂GPU包装已准备：16项CPU集成预检及独立字节、持久计数、来源和有限预算审计PASS。原Torch R、共享S0关闭证明复用、共享S1开启复用沿用原计时方法，同次全部27query、六轮平衡排列。CPU原型21项PASS保持，新增GPU尚未执行；本阶段累计仍1865/2200次、2007.138085/2700秒。完整方案624次/600秒，最坏累计2489次，需用户确认累计前向2200→2500，GPU45分钟不变。

收到新预算明确确认后，保存绑定新plan哈希的回执，执行唯一GPU worker并监控计数、逐位结果、资源和清理；结束后独立CPU复算所有18组原始数据及486份输出，并报告S1对同次S0及原Torch的实际效果。无确认不得启动；短KV稳定优化、prompt共享及真实任务E2E仍待验证。

[完整报告](SHARED-PREFIX-ROWPROOF-GPU-PREPARED-20261012.md)；Knowledge CHAT-SHARED-PREFIX-ROWPROOF-GPU-PREPARED-20261012。

## 131. 短KV三组GPU完成：共享正向，新增行证明优化mixed（2026-10-12T23:31:10+08:00）

短KV288三组GPU对照完成，正式18组/486输出共3,265,920元素逐位等于同次原Torch；S1首query全部30层Q/K/V及attention逐位一致。S0比R中位快14.26%，S1快14.38%，均6/6胜出。但S1比S0仅快0.14%、2/6勝出，flat_or_mixed；不接受此次行证明复用为稳定加速优化。GPU诊断实际scalar30→1，收益只在lease建立首query节省并摊到27次前向。新624次/176.508秒，累计2489/2500次、2183.645901/2700秒，清理及独立审计PASS。

保留现有Torch逐位对齐的共享前缀路径，新行证明复用原型继续私有/默认关闭；不因0.14%mixed信号自动接入或追加GPU测试。下一步建议先用CPU/现有trace查文本prompt投影和cross-attention KV的重复工作，准备独立数值与预算方案，再考虑任务端到端验证。当前累计仅余11次target，完整GPU新实验需要新的有限预算方案；没有预先批准后续实验。

[完整报告](SHARED-PREFIX-ROWPROOF-GPU-RESULT-20261012.md)；Knowledge CHAT-SHARED-PREFIX-ROWPROOF-GPU-RESULT-20261012。

## 132. prompt/cross-KV GPU完成：逐位原Torch，新增共享未获净收益（2026-10-13T01:37:14+08:00）

prompt/cross-KV CPU及GPU验证完成。新版54个正式序列/1512输出/10,160,640元素与同次原Torch逐位一致；三档183个首query GPU投影逐字节哈希、18项GPU失效检查通过。KV288 新P相对S0耗时增加17.33%、1/6胜出、flat_or_mixed；KV1152 新P相对S0耗时增加23.79%、0/6胜出、negative_pilot；KV2160 新P相对S0耗时增加6.48%、2/6胜出、flat_or_mixed。新版三档中位耗时都高于现有S0，当前新增共享不宜接入。初版短/中增时7.56%/7.45%、长档4/6混合及父超时FAIL完整保留。新增3次GPU启动/4152次/2810.664秒；累计6641/7500次、4994.310/18000秒，清理PASS。

继续使用已对齐原Torch的S0；prompt/cross-KV原型保持私有默认关闭。下一步先量化每query守卫的主机耗时，再设计请求生命周期内的变更接口并减少2.502 GiB缓存成本，保留逐位精度门后重新做独立对照。

[完整报告](PROMPT-KV-GPU-RESULT-20261013.md)；Knowledge CHAT-PROMPT-KV-GPU-RESULT-20261013。

## 133. 冻结 S0 阶段成果，完整 episode E2E 方案 CPU 通过（2026-10-13T08:43:56+08:00）

# 投机采样阶段整理与完整 episode E2E 方案

2026-10-13T08:43:56+08:00 | HW chat/research | Asia/Shanghai

当前冻结基线：原 Torch 逐位对齐的 self-attention 历史前缀共享 S0。此前短、中、长 KV 的 verifier 序列已有正向性能证据；最近同次三档中位耗时降低 17.49%、23.42%、34.16%，其中短档 5/6 胜出，仍有波动。这些数据不等于完整 chunk 或任务 E2E 加速。prompt/cross-KV 新增共享虽然精度对齐，但新版相对 S0 分别增时 17.33%、23.79%、6.48%，暂不接入；行映射证明复用也暂不接入。此前报告与失败数据全部保留。

用户明确要求下一步先测完整 episode，trace 长度指真实环境步数。新私有方案使用当前冻结 A_off draft（SHA d55f59c3b467034d21bc21895136bb797c03a752d6934f854ac5e0ee7e031e7c），三个条件 T=原 target、D=原 K2B2 投机、S=D+S0，每条件两个独立 EGL worker 并行跑初态 3/4，共六个完整 LIBERO_10 task0 episode。不同条件串行，不截断 episode，不改阈值/scheduler/CFG/长度策略。

CPU 最终 17 项检查通过：实际生产 loader 30 层、原 CFG packer 值核对、共享 lease 延迟创建、普通前向失效后重建、跨 chunk/异常恢复、推理线程、轻量计数与有限预算、崩溃整份预留收费、真实客户端 CPU import/端口。CPU 小随机权重仅证明接口；先做真实目标完整 chunk 的 D/S 动作/视频、all30 cache/scheduler/VAE/RNG 逐位 GPU probe，通过才启动 E2E。

主计时沿用原 client.elapsed_sec，覆盖环境 reset/稳定、prompt、draft/验证/回退、动作提交、环境步进与 KV 更新；另记录含 EGL 环境构造与 close 的外层 episode wall。服务 bootstrap 和最后输出 drain 不计入任务。所有条件 profiler、teacher、完整 trace、逐步重型 state/RGB observer 关闭，共同轻量 RAM 命令记录；计时后比初始状态、完整 D/S 输出/命令和终态。两个初态只是 pilot，报告配对 T/S 和 D/S，不承诺多任务稳定收益。

成功率不下降、环境步数至少接近原 target 为目标；当前量化参考 S/T<=1.05 是代理对“差不多”的解释。E2E 测量完成后研究轨迹长度控制，当前不先改策略。

资源沿用用户自主执行和 5 小时 GPU 授权。此前累计 6641 intents、4994.309834191576 GPU 秒；新 stage <=5400 秒，累计 <=18000 秒，最多六 episodes/六 EGL 初始化/七 model workers，无隐式重试。每 episode 4000、probe 400 次有限预留，累计前向硬限 35000 由代理在授权内设置，不是用户指定数值；旧零 episode 和旧 budget plan 均未修改。缺失干净终结计数则按整份预留收费。

GPU 启动仍要求 NVML 连续 15 次空闲。单次读数曾报 25% 且显存/compute/graphics 均为零；持久 NVML 采样后实际读数为 0%，原始读数留存于 GPU-readout-before-launch.json，不放宽空闲门。只使用本次精确归属进程，不干预外部任务。

方案：/rjs/cxz/论文探索/投机采样/experiments/speculative_e2e_v1/PLAN.md
CPU gate：/rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v1-20261013-084008/preflight/CPU-GATE.json
冻结 plan：/rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v1-20261013-084008/plan.json
先前结果：/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/PROMPT-KV-GPU-RESULT-20261013.md

当前状态：CPU 完成，GPU probe/E2E 尚未启动，不声称已经有完整任务加速结果。


## 134. 完整episode S0投机正向，轨迹长度基线与累计误差方案（2026-10-13T09:38:15+08:00）

# 完整 episode 投机 E2E 结果与下一步

2026-10-13T09:38:15+08:00 | HW chat / research | zh-CN | Asia/Shanghai

**本次两个受控初态的完整任务，S0 投机相对原 target 加速 1.0817×，总耗时下降 7.55%，属于正向 pilot。** 相对同一冻结 draft 的原投机版 D，共享额外加速 1.0931×。环境轨迹 283/291 步，原 target 293/300 步，合计缩短 3.20%；全部六个 episode 成功。两初态各只有一个样本，不是多任务/多随机种子的稳定收益声明。

## 完整任务结果

T=原 target；D=原 K2B2 投机；S=D 加已对齐原 Torch 的 self-attention 历史共享 S0。使用当前冻结 A_off.pt、global selector、阈值0.007839680526450805。prompt/cross-KV、行证明复用关闭；scheduler/50 action更新/CFG/完整800 horizon 保持原策略。

| 初态 | T秒 | D秒 | S秒 | T/S | D/S | 环境步数T/D/S | 任务结果 |
|---|---:|---:|---:|---:|---:|---|---|
| 3 | 360.235 | 360.407 | 329.583 | 1.0930× | 1.0935× | 293 / 283 / 283 | 全成功 |
| 4 | 360.579 | 367.993 | 336.778 | 1.0707× | 1.0927× | 300 / 291 / 291 | 全成功 |
| 合计 | 720.813 | 728.401 | 666.360 | 1.0817× | 1.0931× | 593 / 574 / 574 | 6/6成功 |

原投机 D 相对 T 合计 0.9896×，本次接近持平、略慢；加 S0 后才取得完整任务的正向收益。速度比是基线耗时/候选耗时，1.0817× 不等于耗时下降8.17%，实际下降7.55%。

主口径为原客户端 elapsed_sec：包含环境 reset/稳定、prompt、所有 draft/验证/回退、动作提交、环境步进和 KV 更新，成功或完整 horizon 才结束。EGL 环境构造/close 的外层 wall 也记录，T/S 合计为 1.0820×，结论一致。模型 bootstrap 和计时结束后的文件 drain 不计入任务；没有按 target 轨迹长度提前截断。

## 时间边界与收益解释

每条件两个独立 EGL worker 同时运行；不同条件分批。profile、teacher、完整 server trace、逐步重型状态/RGB诊断关闭；共同计数在 RAM，无逐前向文件 I/O/CUDA event。实际命令只存 RAM，初末状态和命令在任务计时结束后保存。原模型已有异步输出保存保留在所有条件中。

reset/T5 为共同且不在 S0 中的阶段，T/D/S 每 episode 平均为 116.558/115.097/110.918 秒；跨批约4–6秒的差异不能归因于共享。独立扣除 reset 后，T/S仍为 1.0971×、D/S为 1.1208×；只比较完整 policy 调用，D/S为 1.1317×。因此本次净收益不依赖 reset 变快，但单次两初态与按批顺序的测量仍需扩大样本验证。

init3/4 的 S 提交900/950次 action scheduler更新，其中674/706次为draft；接受深度2分别303/501、312/525棵树。共享对应1026次真实七候选验证。D/S model前向分别959/1010，T各1404；同一初态的D/S调用分布、接受深度和提交数完全一致。

## 精度与轨迹核对

先做真实固定短 KV 完整 chunk D/S GPU probe，共102次 target前向。完整30通道归一化动作、视频、返回动作、30层有效KV及元数据、scheduler、streaming VAE和CPU/CUDA/Python/NumPy RNG均逐位一致。

完整任务中，独立 CPU 重载74对动作/视频 tensor、927,072个元素，逐字节一致；还核对所有保存的观察输入。574条实际执行命令、初末 simulator/controller状态和客户端RNG一致。所有条件同初态的初始状态、初始观察、XML和RNG相同。完整任务没有保存每个时刻的server KV或随机快照，不将短KV probe的缓存检查描述为所有chunk都做过缓存快照检查。

原投机 D 与原 target T 的初始视频相同，但首chunk动作有近似差异；这轮不能把 draft 近似与batch14/batch2数值形状差异分别定量归因。S与D全任务一致，说明这次共享没有额外改变轨迹。具体误差与控制方案见 /rjs/cxz/论文探索/.pipeline/research/speculative-sampling/TRACE-LENGTH-CONTROL-PLAN-20261013.md。

## 编排、旧证据与预算

T两任务在v1完整完成并冻结。进入D时，启动器读取新进程身份失败；之后观察到完整argv与任务匹配，推断为启动窗口的身份竞争。原始错误未保存当时argv，不能声称已经证明当时一定为空。新启动器先登记Popen、最多2秒等待稳定身份、失败清理刚创建且身份未变的PID；CPU验证瞬时空argv、PID变化拒绝、失败清理和5次真实启动。遗漏的本任务客户端按完整argv、任务环境、UID、PGID与start_ticks精确清理；原FAILED.json保留，清理间隔保守计入时间，两个未闭合D预留各4000全额收费。

v2在40GiB初始磁盘门停止，GPU/模型/前向都为0。v3按四新增episode设置1GiB输出硬限、16GiB启动余量与12GiB运行余量，保留所有旧证据；CPU22项通过。T按唯一初态复用已完成数据，D/S在v3补齐；模型、客户端、计时和共享代码与v1逐字节相同。所有失败与跨job复用均保留可追溯证据，未删慢样本或重写旧结果。

最新预算累计 21489 target intents /35000（包含未闭合预留的保守收费）、GPU 6545.956秒，即 1.818小时 /5小时。GPU最终显存0、compute/graphics进程为空、所有本次子进程已退出。生产代码和受保护 state/rules/cycle保持原状，私有原型默认关闭。

## 下一步

当前两个初态已经满足“轨迹至少与原来差不多”。下一步以S0和现阈值为基线，在长尾初态研究**每chunk累计加权误差预算＋夹爪/动作通道约束；预算用尽后剩余denoise走原batch2 target**。完整环境horizon和成功判定不变，以成功、环境步数和E2E联合验收。控制方案属于下一阶段设计，本轮没有启用新策略。

证据：/rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v3-20261013-090832/FINAL_RESULT.json；独立审计 /rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/speculative-e2e-20261013/INDEPENDENT-E2E-AUDIT.json；CPU /rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v3-20261013-090832/preflight/CPU-GATE.json；GPU probe /rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v1-20261013-084008/probe/PROBE-RESULT.json。


## 135. 完整任务耗时归因：后段验证成本与每chunk重复AST编译（2026-10-13T10:19:42+08:00）

# 单 chunk 收益为何没有等比例转化为完整任务收益

2026-10-13T10:19:42+08:00（Asia/Shanghai）｜HW chat / research｜本轮只分析已保存证据，GPU 与 target 新增用量均为 0。

**增时发生在完整 policy/chunk 内部，而不是环境执行或 KV 更新。还有一项已定位的共享接入成本：每个 chunk 重做 30 层 Python AST 处理和编译。** 原投机 D 前段较快、后段较慢，抵消了轨迹变短的收益；S0 已带来净 E2E 收益，但重复安装和共同固定阶段使收益小于 verifier 测试。

先校正比较对象：本轮 T=原 target，D=原 K2B2 投机，S=D 加 self-attention 历史共享 S0。D/T 总时间是 +1.05%，接近持平；S/T 已下降 7.55%，不是完全没有变化。S/D 下降 8.52%，而且两者的完整输出、执行命令、接受路径、轨迹和调用计数一致。这是两个初态、单任务的 pilot；三臂分批运行，不把差异都解释为代码的精确因果效应。

**完整任务的账本闭合**

以下为每 episode 均值，单位秒；实际重载六份原始 episode 与六份 CLOSED 请求账本复算。

| 阶段 | T | D | S | D−T | S−D |
|---|---:|---:|---:|---:|---:|
| reset / prompt / CPU T5 | 116.558 | 115.097 | 110.918 | -1.460 | -4.179 |
| 完整 policy/chunk 调用 | 214.034 | 219.761 | 194.179 | +5.727 | -25.582 |
| 观察与 KV cache 更新 | 10.409 | 9.892 | 9.119 | -0.517 | -0.774 |
| 客户端其余时间 | 19.406 | 19.450 | 18.965 | +0.044 | -0.485 |
| 完整 episode | 360.407 | 364.200 | 333.180 | +3.794 | -31.020 |

客户端其余时间 = elapsed − reset − policy − cache，包含环境 reset/5步稳定、env.step、观察提取及客户端逻辑；没有更细计时，不能把它全部称为“纯环境耗时”。EGL 构造/close 的外层 wall 同样给出 S/T 约 1.082×；模型启动、失败启动清理和任务结束后的文件 drain 没有计入主 episode 时间。

两任务合计，D 的 policy 多花 **11.453 秒**，reset 少 2.920 秒，cache 少 1.033 秒，客户端其余多 0.088 秒，净多 7.587 秒。没有看到环境、KV 更新或传输增加到足以抵消收益。客户端与服务端 policy 累计时差仅 T/D/S=0.249/0.268/0.219 秒，两任务总计；这是传输及请求边界外共同开销的账面差，不能全算成网络时间。

**为什么轨迹短了，D 还是没有更快**

T 有 38 个 chunk，D/S 有 37 个，chunk 数只少 2.63%；环境步数少 3.20%。平均完整 chunk 成本却从 T 的 **11.265 秒**变成 D 的 **11.879 秒，增加 5.45%**。按 T 的平均成本做算术分解：少一个 chunk 约省 11.265 秒，37 个较贵 chunk 约多 22.718 秒，净多 11.453 秒。这是平均值恒等式，不是同状态 chunk 的因果实验。

逐 chunk 记录显示，早期加速不能外推到整个 episode。为避免终局单 worker 尾段干扰，下表仅比较两初态共有的前18个位置；平均值来自服务端原始请求 wall，包含完整 video、action、输出处理。

| 轨迹位置 | T 秒/chunk | D 秒/chunk | S 秒/chunk | D/T 耗时变化 | S/T 耗时变化 |
|---|---:|---:|---:|---:|---:|
| 前段 chunk 1–7 | 10.931 | 10.312 | 10.107 | -5.66% | -7.54% |
| 中段 chunk 8–13 | 11.213 | 11.820 | 10.437 | +5.41% | -6.92% |
| 后段 chunk 14–18 | 11.672 | 14.639 | 11.533 | +25.43% | -1.19% |

这些是各自实际 rollout 的描述性曲线；T 与 D/S 的状态不同，位置也同时关联 KV、接受路径和并发，不证明 KV 是唯一原因。此前同状态固定负载 OFF 测试则直接验证过：短 KV 双需求动作分支快 14.9%–15.3%，长 KV 双需求慢 28.2%–30.2%。这与当前前快后慢一致；旧实验没有 S0，不能冒充当前 S 的同状态测量。

![逐 chunk 曲线与任务阶段耗时](/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/speculative-e2e-attribution-20261013/E2E-ATTRIBUTION.png)

图的左侧使用两初态共有位置的 server wall，右侧使用全部完整 episode 的 client wall；两者口径标明，未混合成同一阶段计时。T 与 D/S 不是逐位置同状态对照。

**验证次数少了，验证工作未按比例减少**

T 每次普通 action forward 是 CFG batch2，D/S 每棵 K2B2 树验证 7 个候选、CFG batch14。两任务 action 分支的前向次数 T=1938、D/S=1122，但累计 CFG 行调用 T=3876、D/S=14556，即 **3.755 倍**。总 target 前向次数少 29.88% 不能当成总计算量少 29.88%；行量也不是精确 FLOPs 或 wall time。

D/S 两任务构建 1026 棵树、执行 3078 次 draft forward、6626 次树 scheduler transition，最终提交 1850 次 denoise 更新。候选打包、分支投影、误差打分、选择/回退会产生额外 CPU 调度和同步。S0 只避免候选历史 K/V 的重复物化/读取，当前 Q/K/V、cross-attention、FFN、video、VAE 仍沿用原计算，prompt/cross-KV 与行证明复用仍关闭。

旧固定负载诊断 ON 中，verifier forward 占 action 69.3%–83.7%，draft model 占 2.2%–4.7%。这支持优先处理验证器成本，而不是先训练或替换 draft。ON 有同步观察扰动，且不是本轮逐阶段计时，不能用这两个范围给当前 E2E 分摊精确秒数；嵌套 tree_build 与 draft/scheduler 不能重复相加。

**共享版新增的明确成本：每个 chunk 重复编译安装**

源码链条：OnlineShared.action 每个 chunk 新建 SharedPrefix 并调用 install；父类 SingleWorkerCHC.install 对30层逐一调用 _private_forward。每次 _private_forward 都重新读取/解析原 model.py、deepcopy AST、校验 AST、compile/exec，最后绑定当前适配器的回调。当前 37 个共享请求都会走这条安装路径，源码推算为 1110 次 AST 构造；原始 E2E 未另加 AST 次数计数器。

| 已核查的范围 | 数据 | 可以得出的结论 |
|---|---|---|
| 当前 D/S 被动日志：chunk 入口至第22个 FRAME START（首次普通动作准备） | D 130.753s；S 170.025s；S 多 39.272s，平均 1.061s/chunk | 共享版在动作入口之前新增了明显主机成本 |
| 同一窗口之后至原 policy 返回 | D 308.500s；S 218.113s；S 少 90.387s | 其后动作/输出段确实更快，前段增时抵消部分收益 |
| 当前主机、真实生产类30层 CPU 安装诊断，6次，无模型 forward | install 中位 1.276s，其中 AST 1.247s，占 97.72% | 重复 AST 编译本身足以产生每 chunk 约一秒的主机开销 |

被动日志窗口包括 video、前置处理和 S 的安装，并非精确 install profiler；两个阶段分批且 CPU 诊断使用随机 tiny 权重，1.276 秒不是 GPU 模型性能，也不能把全部 39.272 秒严格记为 install。**源码、CPU 重现、原始日志三者形成一致证据，重复安装是优先消除的具体成本。** CPU 诊断调用原安装函数，30层原方法与 KV 完全恢复，0模型前向、0 CUDA 初始化；没有改原实验或生产源码。

先前 verifier replay 的 control.install 在计时外；sequence 中冷 lease、staging、key、release 在计时内。“cold setup included”只覆盖 lease 序列的设置，不包含适配器安装。完整在线 chunk 则包含每次安装。先前所谓“单 chunk”须区分：固定 action 分支与 verifier 序列都不覆盖完整 chunk/episode。最新 replay 短/中/长 KV 的 S0 中位耗时下降 17.49%/23.42%/34.16%，也来自单 worker；短档5/6胜出仍为mixed，中/长为positive pilot。当前 episode 是双独立 worker 同卡并行，不能直接沿用那个比例，也不能把中位数改善写成每个chunk稳定加速。

S/D 日志的入口窗口多 39.272 秒，入口后窗口少 90.387 秒，净 policy server 少 51.115 秒。也就是说，看动作入口后的快段会高估净 chunk 收益。窗口是毫秒精度的主机日志重建，涵盖原保存/CPU copy/empty_cache；这些子项仍未在本轮单独计时。

**固定阶段继续稀释 E2E 收益**

reset/prompt/CPU T5 每 episode 约111–117秒，占 T 完整时间32.34%，S0 没有优化它。D→S 完整 policy 实测下降11.64%，policy占 D 总时间60.34%；只让 policy 发生这一变化，其他阶段保持 D 原值，算术上只带来约7.02%的 E2E 降时。这是固定耗时的直接稀释作用。

本次 S/T 的实际7.55%中，还包含非共享阶段 reset 的跨批变化。把 reset 算术统一成 T 的耗时，S/T 降时是 **5.99%**，仍正向；这不是新测试或精确因果收益，只是避免把 T5 波动归功于共享。原报告7.55%作为观察到的任务结果保留，不改历史。

**下一步：先消除重复编译，暂缓轨迹控制**

建议先做私有的“编译模板复用”：按冻结源码 SHA/原 forward 身份缓存不可变的 AST 编译结果，在进程内复用；每个 chunk 仍重新绑定当前 callback/owner，并独立建立、撤销 lease。不能跨请求复用带 model/KV/旧 callback 的函数，也不能跳过源码身份、30层准入、CFG 行、cache generation、版本和异常失效检查。首 chunk 的冷编译也应计入完整 chunk 时间，不能通过移到计时外制造收益。

先 CPU 验证原有所有失效/线程/恢复约束、完整输出及安装成本；然后以新有限方案做同状态短/中/长 KV、同卡双 worker 的 T/D/S/模板复用 S 对照。正式性能使用 profiler OFF，至少分别记完整 chunk、video、action入口设置、action、后处理，诊断 ON 单独运行；完整30通道动作、video、KV元数据、scheduler和RNG保留逐位门。通过后再复测完整 episode。

如果这项接入成本消除后长段仍无净收益，再拆 batch14 验证器内 self/cross attention、FFN、打包与同步的成本，研究计算复用或成本准入。reset/T5 缓存可作为后续三臂共同工程基线，但不能记作投机专属收益。当前不改阈值、树策略或环境轨迹，不重新训练 draft；原轨迹控制方案保存，优先级按用户最新要求后移。

本轮完成的是原因分析与下一步设计，没有声称已实现或验证新的 GPU 优化。新 GPU/target 用量=0；累计仍21489 intents、6545.956473秒/18000秒。protected state/rules/cycle、公共代码及旧结果由归档核验，HW仍为研究追加通道，不重开M04。

**可追溯证据**

- [本轮复算与完整逐 chunk 数据](../../../review/OMISSIONS.md)；[CPU安装诊断](../../../review/OMISSIONS.md)；[分析工作清单](../../../review/OMISSIONS.md)。
- [原完整episode结果](SPECULATIVE-E2E-RESULT-20261013.md)；[旧同状态负载矩阵](FIXED-LOAD-COST-RESULT-20261011-213036.md)。
- [每chunk安装入口](../../../投机采样/experiments/speculative_e2e_v3/online_shared.py)；[30层安装循环](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)；[AST处理与编译](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)；[replay计时边界](../../../投机采样/experiments/shared_prefix_gpu_v3/worker.py)。
- [原始action候选验证与计数](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)；[CPU复制后异步文件保存](../../../投机采样/wam-p0-worktree/wan_va/utils/utils.py)。


## 136. 完整chunk开销优化顺序与逐项收益检查（2026-10-13T11:09:33+08:00）

# 投机＋共享 KV：逐项消除完整 chunk 开销

2026-10-13（Asia/Shanghai）｜HW chat / research｜状态：计划已拟定，优化实现与新 GPU 测量尚未执行。

用户目标：减少吞掉共享收益的接入、同步和候选组织开销；逐项比较，只有完整 chunk 与最终任务有收益的改动才保留。环境轨迹控制继续延期。

## 1. 先区分两种“树”

- Python AST 是源代码的抽象语法树。当前适配器解析原 `WanAttention.forward`，只替换候选历史注意力片段，再生成 Python 函数。这与投机候选树无关。
- K2B2 验证器有固定 7 行：`root, first[0], first[1], leaf[0,0], leaf[0,1], leaf[1,0], leaf[1,1]`。CFG 后是 14 行。当前已用 tensor 保存分支状态，再通过 `stack/cat` 打包，并非动态分配七个指针树节点。
- 用固定七行 tensor 缓冲区能减少部分打包与复制；它不会消除 AST 处理，也不会消除三次 draft、六条候选边的 scheduler 计算和一次 batch14 target 验证。

## 2. 已有证据与正确的基线

T=原 target；D=未共享 K2B2；S0=D 加历史 self-attention KV 共享。旧短/中/长测试主要比较同 batch14 的候选验证器，不含在线 `control.install()`，不是完整 chunk 对 T 的证明。

CPU 六次真实生产类、30层、随机小权重安装诊断：install 中位 1.276045 秒，AST builder 中位 1.246975 秒，占 97.72%；无模型前向、无 CUDA 初始化。它证明主机设置开销，不是 GPU 阶段的精确计时。

保存的两个初态在第17、18个 chunk 中，S0 对 T 完整请求慢 2.32%–7.69%，对 D 快 19.28%–26.06%。四个 chunk 均值：

| 被动日志范围 | T 秒 | S0 秒 | S0−T 秒 |
|---|---:|---:|---:|
| chunk 入口至首次普通动作准备 | 3.597000 | 4.688250 | +1.091250 |
| 首次普通动作准备至原请求返回 | 8.140802 | 7.652433 | -0.488369 |
| 完整 policy/chunk 请求 | 11.737802 | 12.340683 | +0.602881 |

更细的原日志拆分：前20次视频迭代跨度 T=3.446000 秒、S0=3.379250 秒；最后一次视频准备至首次动作准备 T=0.150500 秒、S0=1.308000 秒。主要新增等待落在适配器安装所在的入口区间。该区间仍包含最后一次视频 forward，不能全记成精确编译耗时。T 与 S0 的 rollout 状态不同、三臂分批运行；这些是实际曲线的描述性证据，后续收益用同状态对照确认。

现有完整 episode 两初态 pilot：S0/T 总耗时下降7.55%。本计划目标是进一步降低完整 chunk 和任务耗时；不把去除约一秒 CPU 成本直接承诺为20%的任务加速。

## 3. 按增量收益决定是否保留

| 顺序 | 改动与范围 | 检查结果 | 性能检查与决策 |
|---|---|---|---|
| O1 优先 | 复用已编译 Python forward 模板；每个进程/源码版本构建一次，30层及后续 chunk 复用代码对象；每次绑定当前回调 | 旧 S0 与 O1 的候选输出、完整动作/video、接受路径、KV状态和 RNG 一致；异常退出后正确恢复 | 冷首次设置与后续设置分别记录；CPU 热安装目标低于100ms、热路径 AST 构建次数为0；长 KV 完整 chunk 比旧 S0 更快才保留 |
| O2 按残余占比推进 | 预先取 timestep/sigma 的 CPU 值；保留六条边原 norm 计算，只合并标量回读；将固定布局校验移出重复调用，合并动态有限值检查 | 保持 scheduler 次序、六条边误差、阈值比较、tie-break、回退与接受路径；KV变更仍使旧共享内容失效 | 先诊断同步成本，再对 O1 与 O1+O2 做长 KV 对照；未稳定胜出就关闭，不因同步次数少而认定加速 |
| O3 按残余占比推进 | 固定七行 tensor 工作区；减少重复 `verifier_states()` 拼接、`stack/cat/clone` 和打包；固定索引取 root/internal/leaf | 比较7行顺序与 timestep、分支状态、输出；覆盖全部接受深度及回退，确认重用缓冲区不会覆盖后续仍需使用的状态 | 对当前已保留版本与加 O3 的版本比较；保留可重复的完整 chunk 收益，收益混合则不加入默认路径 |
| O4 汇总验证 | 只组合已保留的项目，补齐短/中/长完整 chunk，复测 episode | 最佳新版本与旧 S0 的任务成功、实际执行动作、终局状态、轨迹及接受路径一致 | 同时报告对 T 的总收益与对旧 S0 的增量收益，报告冷启动与任务全程成本；只按实测结论推广 |

源码校验、原 forward 可执行结构的重复编译也在 O1 的成本范围内；可以缓存不可变结果，按源码/原函数身份绑定。先保留当前每 chunk 的回调和共享失效生命周期，避免在 O1 同时改成跨 chunk 的持久 KV 缓存。常驻适配器仅在 O1 后残余安装仍值得处理时另列一项，不与首次模板优化混测。

只缓存不可变代码模板；不缓存绑定旧 controller/KV 的函数，不跨请求保留历史计算结果。动态 cache/行映射/普通 target 写入后的失效检查仍有正确性作用；优化其重复工作，不通过关闭它们制造收益。已有 prompt/cross-KV 与行证明实验缺乏增量收益，不自动并入本轮。

## 4. 每项怎么测

1. **私有 CPU 实现与预检。** 使用新目录，旧实验、公共模型、checkpoint 与已有结果保持冻结。O1 先验证冷/热设置计数、两个 controller 的回调隔离、真实30层安装/恢复、异常清理与旧 S0 输出一致性。只覆盖此次改动影响的约束，复用已有数学和失效检查。
2. **先做长 KV 同状态完整 chunk。** 原 T、旧 S0、新 O1 三臂；相同 snapshot、请求、seed、scheduler、prompt；两独立 worker 同卡并行，六轮三臂顺序平衡。每臂首次冷 chunk 单列并保留，它也充当后续预热；随后六轮全部保留，共42个完整 chunk。定时包含 video、在线安装、draft、candidate pack、verify、选择/回退、动作提交、原后处理至返回。snapshot 恢复位于每次请求前，不计入自然请求耗时；恢复成本另记。
3. **诊断与正式性能分开。** 正式结果 profiler OFF；诊断 ON 只用于定位 video、安装、draft/tree、verifier、评分/选择、后处理占比，嵌套区间不能重复相加。共用诊断附加成本不用于宣称性能收益。
4. **逐项加法。** O2、O3 都与已经保留的版本成对比较；优先长 KV，同状态、两 worker、至少四轮 AB/BA 平衡。不能仅靠旧 D 或微基准认定新版本有收益；没有可重复收益的项目保持关闭。
5. **最终短/中/长覆盖。** 短 KV288、中 KV1152，最佳版本对旧 S0；两 worker、四轮 AB/BA，加每臂一个单列冷样本，共40个完整 chunk。长 KV2160沿用对应已接受版本的对照，若组合有变则补新的长档。记录 fixture 的实际 KV 元数据，不能只按轨迹位置命名长档。
6. **最终完整任务。** 同一个 LIBERO 任务、初态3/4、旧 S0与最佳新版本，共4个完整 episode，按配对顺序运行，每臂两 EGL worker。计入 reset/prompt、draft、验证、回退、提交、观察/KV更新、环境执行；启动加载和任务计时分别报告。保留原 T 的历史 pilot 作为参考，不将跨批 T 比较包装成同批因果对照。

完整输出一致性比较对象是“旧 S0 vs 工程优化后的 S0”。原 T 不做投机，其结果与 S0 本来可能不同；不能错误要求二者完整轨迹逐位一致。候选验证的数学仍用原 Torch 输出作参照。

性能汇报包括完整请求 wall、中位数、每轮/worker 差异、冷首次成本、异常与全部样本；稳定改善才采纳，混合结果先不加入。阶段目标是 O1 尽量收回当前入口中约一秒的重复工作，实际收回量以 GPU 完整请求为准。最终必须同时看完整 chunk 与任务成功/全程耗时。

## 5. 范围与资源

- 本轮只拟定计划、复算保存的日志并追加 HW；新增 GPU启动、target前向与GPU秒数均为0。O1/O2/O3 尚未实现，新的加速结果尚未验证。
- 沿用用户已批准的累计 GPU 5小时（18000秒）、累计 target intents 35000。当前累计21489 intents、6545.956473秒，余13511 intents、11454.043527秒；不修改旧 GPU plan、旧一次性授权或旧失败记录。
- O1 首轮长档建议新有限方案：最多3200 intents、一次启动、15分钟，正式42个 chunk最多3024 intents，余量覆盖同一方案内限定检查；无自动重试。
- 最终短/中覆盖建议上限3000 intents、15分钟；最终4 episode每个最多1800 intents，总上限7200 intents、30分钟。三个预估上限合计13400，累计34889，仍在35000内；具体可执行 plan 必须在实际 CPU 通过后按入口/诊断/完整请求的真实计数再次核算。
- O2/O3 的 GPU 增量对照只使用已闭合前序实际用量释放出的余量，每次生成独立有限方案后再执行；没有足够余量则不启动该项，先保留已确认有利的 O1。不能提前假定保守预留都会退款。
- 正式采集遵守现有 EGL＋至少两独立 eval worker、GPU/端口/master端口/provenance 检查；已有产物不覆盖。GPU 时间计入该次加载到清理；暂停/中断的已发生消耗仍入账。
- HW仍沿用研究/chat追加通道；不新建 Cycle、不重开M04、不修改 state/rules/cycle。训练、接受策略、阈值与环境轨迹控制不列入这次工程优化。

## 6. 当前工作清单

- [x] 确认代码 AST 与7节点候选验证树的区别。
- [x] 依据原日志与 CPU 诊断拟定优先级、比较对象与保留条件。
- [x] 明确首项 O1 的实现边界、完整 chunk 检查与资源方案。
- [ ] O1 私有实现、CPU 输出/生命周期/冷热成本验证。
- [ ] O1 同状态长 KV 完整 chunk GPU 对照。
- [ ] 按残余开销决定 O2/O3 是否值得尝试，逐项验证并保留。
- [ ] 最佳组合短中长完整 chunk、完整 episode 与结果汇报。

## 证据

- [已完成耗时归因](SPECULATIVE-E2E-ATTRIBUTION-20261013.md)；[原完整任务结果](SPECULATIVE-E2E-RESULT-20261013.md)。
- [CPU安装诊断](../../../review/OMISSIONS.md)；[逐chunk数据](../../../review/OMISSIONS.md)。
- [AST构造](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)；[每chunk安装](../../../投机采样/experiments/speculative_e2e_v3/online_shared.py)。
- [当前固定7行结构](../../../投机采样/scripts/p0_batched_verify.py)；[树状态构造](../../../投机采样/scripts/p0_batched_verify.py)；[六条边评分回读](../../../投机采样/scripts/p0_batched_verify.py)；[每树timestep/sigma回读](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)。


## 137. O1模板复用完整长KV正向与采集修复（2026-10-13T13:09:51+08:00）

# O1 编译模板复用：完整长 KV chunk 正向

2026-10-13T13:09:51+08:00（Asia/Shanghai）｜HW研究追加｜只完成第一项O1；O2/O3、长度覆盖与新episode测量待后续执行。

**结论：保留O1为下一轮实验基线。固定长KV2160、两个初态、同状态双worker、六轮平衡三臂完整chunk对照中，O1对旧共享版降时12.81%/12.87%，对原target降时11.11%/11.61%；两种比较在两个worker均6/6胜出。**

T=原target；S0=原K2B2投机＋历史KV共享；O1=S0加不可变Python编译模板复用。本轮不是只测候选verifier，完整计时包含video、每chunk安装、draft、候选组织、验证、选择/回退、提交、原输出保存/CPU拷贝与返回。所有冷样本和六轮主样本保留。

| 初态 | T完整chunk中位秒 | 旧S0秒 | O1秒 | O1对S0降时 | O1对T降时 |
|---|---:|---:|---:|---:|---:|
| 3 | 11.973 | 12.206 | 10.643 | 12.81% | 11.11% |
| 4 | 12.029 | 12.203 | 10.632 | 12.87% | 11.61% |

旧S0对T在两个worker分别慢1.95%/1.44%，而O1变为明确更快。改善来自接入成本消除，没有修改候选节点、接受规则、阈值、scheduler、模型权重或历史共享的注意力计算。

## 实现与设置成本

O1按源码/原forward代码身份缓存不可变代码对象，原AST构造仍用于首次模板生成。30层与后续chunk复用代码；每层/每chunk新建函数namespace，绑定当前controller回调。源码、全30层真实class/forward/SDPA准入、KV失效与租约生命周期仍执行。缓存没有model/tensor/controller/旧callback/函数globals。

原安装和准入函数的可执行代码保持原样，只在私有namespace中替换构建器与不可变原代码结构的缓存助手；旧源文件和类globals不修改。CPU真实生产类随机小权重验证13项通过，涵盖完整30层候选输出、普通写入后重开lease、两chunk、回调隔离、弱引用回收、暖缓存拒绝坏源码/末层backend、部分安装异常和清理、真实推理线程。首轮GPU集成11项，新采集版13项通过。

| 安装范围 | 旧S0秒 | O1秒 |
|---|---:|---:|
| CPU真实30层、同主机中位 | 1.045927 | 热 0.015449；冷 0.054495 |
| GPU进程0六轮热安装中位 | 1.626306 | 0.019602 |
| GPU进程1六轮热安装中位 | 1.617746 | 0.017727 |

两个GPU进程O1首次安装分别0.087065/0.194513秒；后续0次AST构造、每chunk30次新绑定。对照中安装省约1.60–1.61秒，完整chunk净省约1.56–1.57秒，数量吻合。它支持重复安装是主要增时来源，不将分别取中位的差值当成精确逐段分摊。CPU tiny fixture不是GPU模型的性能证明。

## 输出、完整状态与统计复核

42个完整chunk、6个冷样本＋36个主样本全部保留，每worker21个。共享两臂全部动作30通道、video、物理动作、接受路径/计数、全30层有效K/V字节及完整cache元数据、scheduler、VAE状态与RNG一致。无target失败，两个worker各1246次host-returned target意图；总2492。T与投机的完整动作本来可能不同，正确性比对对象是旧S0与O1。

全状态冷归档独立CPU重新读取，比对440对tensor、3281448092字节、4对numpy数组，逐字节一致；0 CUDA初始化、0新增GPU。六轮三臂顺序遍历全部排列；各臂/worker主耗时MAD/中位均低于7%，未删除样本。数据限定为两个真实保存状态的固定long-KV pilot，没有将它称为新完整episode加速比。

## 首轮采集失败与修复

v1在900秒窗口内完成34/42个chunk；计时外重复回传、有限值检查和SHA计算全1.6GB状态造成采集过慢。到达退出边界后，报告dict有重复stage字段，终端失败报告没有正常写出。所有已完成样本和原代码/plan保留；GPU进程已清理、无数值失败证据。

v2保持O1推理核心源文件逐字节相同，仅修复采集和报告：GPU逐字节比较每次恢复与结束的全30层有效K/V、完整mask/id/is_pred，较小的字段/VAE/scheduler/RNG仍回传并哈希；冷S0/O1保留完整CPU状态归档。输入与结束GPU参考常驻内存，对所有正式三臂共同保持；比较与额外归档均在完整request定时之外。正式42个chunk在440.511651秒完成，证明采集修复有效。该额外参考内存是实验共同开销，真实episode复测仍需确认。

v1没有durable CLOSED计数，按两worker完整预留保守计3200，RAM已观察2030仅作下界，不退款。因退出报告未落最终时间，按启动到确认进程消失的1702秒上界保守计账（含其间空闲等待，不冒充精确GPU消耗）。v2确切2492次、440.511651秒；两次新任务合计保守5692次、2142.511651秒。

当前累计27181/35000 target intents、8688.468124/18000 GPU秒。v2终态NVML分配内存0、compute/graphics为空、全部本任务root退出。protected state/rules/cycle、公共代码、旧证据保持不变。

## 下一项与范围

O1已确认有利，作为后续最好私有基线，默认关闭的生产边界保持。下一项O2先查看每树timestep/sigma回读和六条误差标量同步，保留各条norm计算与选择次序，尝试一次批量回读；不混入七行工作区。只有对O1有可重复完整chunk增量收益才保留。O3按残余占比推进。

后续短/中覆盖与完整任务按实际闭合剩余预算重新分配有限方案。原计划预留7200的四episode上限与其他全部预留已不能同时无条件满足余7819次，不能照搬旧预算；优先使用已确认的每状态/episode计数制定更紧的上限，并保留失败消耗。当前不声称O1已提高新episode总耗时或改变轨迹长度。

## 文件

- [完整逐次数据与统计](../../../review/OMISSIONS.md)；[独立归档字节检查](../../../review/OMISSIONS.md)；[CPU13项结果](../../../review/OMISSIONS.md)。
- [模板实现](../../../review/OMISSIONS.md)；[GPU状态字节检查](../../../review/OMISSIONS.md)。
- [新GPU方案](../../../投机采样/results/runtime/p0-chunk-template-v2-20261013-125000/plan.json)；[完整GPU关闭账本](../../../投机采样/results/runtime/p0-chunk-template-v2-20261013-125000/GPU-CLOSED.json)。
- [原优化顺序计划](CHUNK-OVERHEAD-OPTIMIZATION-PLAN-20261013.md)；[失败首轮的保守核对](../../../投机采样/results/runtime/p0-chunk-template-v1-20261013-115600/RECONCILED-FAILURE.json)。


## 138. O2标量回读完整长KV增量对照（2026-10-13T13:59:39+08:00）

# O2 标量回读与 scheduler timestep 往返：完整长 KV 对照

2026-10-13T13:59:39+08:00（Asia/Shanghai）｜HW研究追加｜第2项O2完成，O3与新episode尚未执行。

**本轮两个worker均有重复增益，暂保留 O1＋O2 为最佳私有基线。** 以下为 O1 对 O1＋O2 的增量结果，不能把第1项O1的约12.8%收益算作O2收益。

| 初态 | O1完整chunk中位秒 | O1＋O2秒 | 耗时下降（负值=变慢） | O2逐轮胜出 |
|---|---:|---:|---:|---:|
| 3 | 10.823955 | 10.648761 | +1.619% | 4/4 |
| 4 | 10.810641 | 10.694027 | +1.079% | 3/4 |

固定KV2160、同一源snapshot/请求/frame56，两个独立model worker和两个真实NVIDIA EGL环境并行。每臂1个冷样本＋四轮AB/BA交替，共20个完整chunk（4冷、16主）；全数保留、无重试。判断条件为两个worker各自中位及均值都下降，且至少3/4配对轮次胜出。数据量为固定状态pilot，不能推广为所有长度或任务的加速保证。

## 改了什么，为什么要改

1. 原action scheduler的timesteps/sigmas由CPU torch.linspace生成，并在整个动作循环内保持不变。O2在每次动作helper内部一次取表，然后原三元素sigma_path/target_times改为Python索引。这里主要省Python标量拆解，原本就不是六次GPU传输；每次请求重读，不跨请求缓存表或KV。
2. 原六条边分别计算 `predicted.float()-target.float()`、原channel选择、flatten、`torch.linalg.vector_norm` 和原scale乘法。O2保留六次独立计算及次序，仅把已算出的六个float32标量stack后一次 `.cpu().tolist()`，再构造原Python float成本和原scores；global/greedy tie-break及threshold比较仍原实现。
3. 原候选边scheduler callback先创建GPU float32 timestep；原 `FlowMatchScheduler.step` 立即把它 `.cpu()`，再从CPU时间表查找sigma。O2直接构造值相同的CPU float32 timestep，调用未修改的原scheduler，省GPU分配及搬回CPU。scheduler的查找、步进、投影、回退次序和精度均保留。

本fixture每chunk 29棵树：误差回读由174次变为29次；每树6次CPU timestep/sigma拆解改为每动作请求2次整表读取；192次scheduler timestep的GPU→CPU往返被取消。六个norm仍独立，不做跨行向量化归约。原九处树构建动态finite检查、源码/30层/lease/cache失效守卫、固定七行状态拼接及候选打包都保留，因此仍有同步和pack成本。

私有unbound action/score模板在进程准备阶段各编译一次，其prepare耗时分别为0.107531,0.112560秒，不计入每次request计时，但计入本次GPU加载到退出总耗时；全部20次request内无再次编译。无公共源码或旧API globals修改。

## 输出一致性与运行范围

CPU9项通过：六成本float32/bfloat16精确相等，channel/axis、tie-break、threshold紧邻边界、接受深度0/1/2、NaN/Inf与形状错误、每请求表刷新、真实生产30层tiny target＋冻结训练draft完整动作helper（拒绝/接受/尾部）、KV/scheduler/RNG、异常恢复和默认关闭。AST修改逆向还原后与原可执行结构相同，只允许上述具体读取及临时timestep设备改动。CPU首次fixture漏设draft模式导致三项失败，修正fixture配置后通过；两个CPU日志均保留，未产生GPU重试。

GPU20个完整chunk的动作30通道、video、物理返回值、选择/接受计数、全30层有效K/V字节和完整id/mask/is_pred元数据、scheduler、VAE和RNG均与O1一致。冷档完整CPU归档独立重新读取，440对tensor、3281448092字节、4对numpy数组逐字节一致；CPU归档复核无CUDA初始化。GPU每次热档结束和每次恢复均比较全30层有效字节，不能仅靠最后一层或动作误差宣称一致。六条具体误差逐位比较覆盖CPU；GPU覆盖完整最终输出、全部接受计数和状态。

主计时为原Runtime完整request wall，包含video、O1在线安装、draft/树构建、target verify、误差回读、选择/回退、提交和原保存/CPU拷贝至返回。原输出plumbing保留。额外restore/全状态精度比较在timer外，对所有正式臂保留相同GPU参考驻留；这仍是带共同实验参考内存的同状态chunk对照，尚非自然episode。

以下被动host窗口辅助观察方向，profiler OFF，没有在中间插入额外CUDA同步；不能把它当精确GPU kernel分摊，也不能把分别中位差相加当主耗时。

| 初态 | O1 video窗口秒 | O2 video窗口秒 | O1 action窗口秒 | O2 action窗口秒 |
|---|---:|---:|---:|---:|
| 3 | 3.836624 | 3.781507 | 7.015194 | 6.864025 |
| 4 | 3.795199 | 3.828030 | 7.003820 | 6.809855 |

冷样本保留如下。O1先运行且承担本进程attention模板首次构建，O2冷样本复用该模板；正式收益结论用四轮热对照，不靠有顺序差别的冷样本。

| 初态 | 臂 | 冷完整chunk秒 | O1模板安装秒 |
|---|---|---:|---:|
| 3 | O1 | 11.672134 | 0.073463 |
| 3 | O2 | 10.712254 | 0.018942 |
| 4 | O1 | 11.715856 | 0.076369 |
| 4 | O2 | 10.656197 | 0.018384 |

## 资源与下一项

本次GPU启动1次、model启动2次、EGL初始化2次，精确1060次target intents，无失败前向。两个worker各10个完整chunk和durable CLOSED计数均写出。退出时supervisor只等2秒，随后现有归属守卫拒绝清理，原FAILED.json保留；没有修改guard或重跑GPU。独立CPU检查随后确认四个root全部不存在、三次NVML分配0且compute/graphics为空，重新检查全部20个DONE与两个530次CLOSED后，另存RECONCILED-CLOSED.json。没有对任何进程发信号。精确竞态原因未取到当时proc快照，不能断言已经定位；下轮GPU前应优先延长有界自然退出等待并测试终端进程收尾，保留所有活进程归属检查。

原supervisor退出报告记录301.990546秒；为涵盖报告后进程自然退出，按加载开始到首次确认全部消失的461.008183秒上界保守计账（包含部分空闲确认等待），不是精确kernel/占用时间。完整20个chunk正确性与性能数据不受退出阶段报告失败影响，未新增前向或GPU启动。

累计28241/35000 intents、9149.476307/18000秒；余6759 intents、8850.523693秒。累计仍包含O1旧失败的保守完整预留和时间上界，不退款。下一项和最终短/中/完整episode必须按剩余真实预算重算，不能照搬原7200次episode预留。

按原顺序进入 O3 的CPU阶段：固定七行tensor工作区，减少重复verifier_states/stack/cat与有效状态复制；与已保留最佳版本逐项对照。先据剩余预算和CPU残余成本决定有限GPU方案，不动阈值、权重或轨迹。

这次没有新短/中完整chunk、没有新完整任务E2E加速比；轨迹长度仍延期处理。O2通过正确性不等于一定快，采纳以本轮完整chunk比较为准。public/state/rules/cycle和旧结果保持不变。

- [全部20次数据与统计](../../../review/OMISSIONS.md)；[CPU检查](../../../review/OMISSIONS.md)；[独立冷档字节复核](../../../review/OMISSIONS.md)。
- [O2私有实现](../../../review/OMISSIONS.md)；[新GPU方案](../../../投机采样/results/runtime/p0-chunk-scalar-v1-20261013-133800/plan.json)；[GPU清理与保守计账](../../../投机采样/results/runtime/p0-chunk-scalar-v1-20261013-133800/RECONCILED-CLOSED.json)；[原退出报错](../../../投机采样/results/runtime/p0-chunk-scalar-v1-20261013-133800/FAILED.json)。
- [已完成O1报告](CHUNK-TEMPLATE-O1-RESULT-20261013.md)；[原逐项优化计划](CHUNK-OVERHEAD-OPTIMIZATION-PLAN-20261013.md)。


## 139. O3七行工作区完整长KV增量结果与退出修复（2026-10-13T15:02:36+08:00）

# O3 固定七行工作区与批量通道搬运：完整长KV结果

2026-10-13T15:02:36+08:00（Asia/Shanghai）｜HW研究追加｜O3完成；新完整episode尚未复测。

**本轮两worker均有重复增益，暂保留O3，最佳私有基线为O1＋O2＋O3。** 已有最佳O1＋O2简称O2，新臂O3为O1＋O2＋O3；下表只计本项增量。

| 初态 | O2完整chunk中位秒 | O2＋O3秒 | 耗时下降（负值=变慢） | O3逐轮胜出 |
|---|---:|---:|---:|---:|
| 3 | 10.437381 | 10.253499 | +1.762% | 4/4 |
| 4 | 10.471006 | 10.232302 | +2.280% | 4/4 |

固定KV2160、同一snapshot/请求/frame56/seed/scheduler，两个独立模型worker和两个真实NVIDIA EGL环境并行。每臂1冷＋四轮AB/BA，共20个完整chunk（4冷16主），全部保留；采纳条件为两worker中位及均值都下降且至少3/4轮胜出。本轮为固定状态pilot，不推广为全部长度或任务保证。

## 实现与边界

- 动作请求内复用7行state、6条direction、previous-direction缓冲区。节点顺序仍root/first0/first1/leaf00/leaf01/leaf10/leaf11；原三次draft、六条scheduler边、shape/finite检查次序和算术不变。
- 原first/second/leaf stack写入预分配输出，去掉其末尾三个clone；root和direction仍在原return位置复制。pack/score复用同一7行view，移除每树两次verifier_states的cat。
- 候选full latent由七次zeros/index_copy加cat改为一次zero和批量index_copy；target compact由七次index_select加stack改为一次批量index_select。请求内复用通道索引，不引入新浮点归约或近似。
- 六条norm、scale、threshold、tie-break、接受/回退、terminal target以及KV失效、owner/lease沿用O2；prompt/cross-KV与rowproof仍关闭。模型、训练和轨迹规则未改。
- buffer只存活于当前动作请求。下一树覆盖前树时，旧tree.verifier_states显式失效；execute返回状态保持原clone。全trace或teacher开启时自动使用O2，保留历史快照语义，不跨chunk保留buffer。

本fixture每chunk29棵树，新臂实际仅分配5份工作存储（state/direction/previous/full latent/channel index）、通道索引上传一次；七行view读取58次，batch scatter/compact各29次，结束后全部关闭。模板仅在进程准备时编译，热路径不解析/编译。O1/O2核心与既有CUDA共享算子字节相同。

## 精度与计时

CPU14项通过：全部tree字段/七行次序、callback次序/RNG、六成本/route/阈值及接受深度、批量scatter/compact/padding覆盖/通道顺序变更、buffer地址复用与失效、输出独立存储、真实生产30层tiny target加冻结训练draft的完整动作helper/KV/RNG、异常清理、默认关闭和全trace/teacher回退。首次teacher测试记录器不能序列化dataclass，修正fixture后通过，两个CPU日志保留。

GPU20个完整chunk的30通道动作、video、物理返回、接受计数、全30层有效K/V字节及全部mask/id/is_pred元数据、VAE/scheduler/RNG均与O2一致。每次恢复和结束都比较有效KV全字节；较小状态/输出每次hash。独立CPU复核440对tensor、3281448092字节、4对numpy数组一致，0 CUDA初始化。

为保留已有数据并控制磁盘新增量，新cold O2与旧冻结O2完整状态hash对齐并复用旧归档；只新增两份cold O3完整归档，再独立读取旧O2与新O3逐字节比较。正式性能仍用本次同批O2/O3交替实测，未拿历史O2耗时作基线。磁盘入口18GiB、运行保留12GiB、新输出cap4GiB。

主计时为原Runtime完整request wall，包含video、安装、draft/tree、verify、score/选择/回退、提交以及原保存/CPU拷贝至返回。原输出plumbing保留。额外restore/精度比较在timer外，两臂共同驻留同样GPU参考；冷样本保留但不作热增量结论。以下被动host窗口没有额外中途CUDA同步，不是精确kernel分摊。

| 初态 | O2 video秒 | O3 video秒 | O2 action秒 | O3 action秒 |
|---|---:|---:|---:|---:|
| 3 | 3.673322 | 3.610699 | 6.756308 | 6.657623 |
| 4 | 3.718483 | 3.621627 | 6.765143 | 6.618833 |


## 退出和CPU预检修复

退出改为全root共用10秒自然等待；终端root用poll/reap，仍存活且PID/start/UID/PGID/argv精确匹配、组内所有活进程归属正确才允许信号，TERM/KILL也有共同截止时间。6项CPU回归覆盖真实2.2秒子进程自然退出、精确归属、身份变更、zombie/活empty-argv和终端状态竞态。实际本次清理发送信号0次，全部root成功回收。旧O2退出失败及保守消耗保持不变。

首个O3 v1集成预检保留了虚拟6GiB样例，新cap4GiB，原disk guard正确拒绝；该方案0 GPU启动/前向。v2仅修正fixture为3GiB合法样例并新增6GiB拒绝检查，O3/退出核心与已通过CPU14＋6项的v1字节相同，旧v1源码/plan/失败记录冻结。新方案生成时逐份SHA验证旧输入，后续入口继续校验关键源码/算子SHA及所有文件size/mtime/ctime/inode/device变更守卫；测量后再次全SHA复核。未放松活GPU归属检查或重试GPU。

## 资源与下一步

本次GPU启动1次、target intents精确1060，加载至清理273.687741秒。两worker各530次、失败前向0；root已退出，终态NVML分配0、compute/graphics为空。累计29301/35000 intents、9423.164048/18000秒；余5699前向、8576.835952秒，仍包括旧失败保守消耗。

进入O4，优先以O1＋O2＋O3对旧S0做两初态、四完整episode同批E2E对照：报告总耗时/加速比、成功、环境步数和全部输出一致性。随后用真实剩余前向补短/中完整chunk；不先做更多细碎优化，轨迹控制继续延期。

四episode按旧冻结实际计数收紧有限上限，不沿用旧7200次预留；完整任务计入reset/prompt、draft/验证/回退、提交、观察/KV更新和环境执行，同时检查成功和环境步数。本次长KV chunk结果不能作为新E2E加速比，轨迹控制继续后置。

- [全部逐次结果](../../../review/OMISSIONS.md)；[CPU14项](../../../review/OMISSIONS.md)；[退出CPU6项](../../../review/OMISSIONS.md)；[独立字节检查](../../../review/OMISSIONS.md)。
- [O3实现](../../../review/OMISSIONS.md)；[GPU方案](../../../review/OMISSIONS.md)；[GPU清理计账](../../../投机采样/results/runtime/p0-chunk-rows-v2-20261013-144300/GPU-CLOSED.json)。


## 140. O4优化组合完整LIBERO端到端结果（2026-10-13T16:20:41+08:00）

# O4：O1+O2+O3 完整 LIBERO episode 端到端结果

2026-10-13T16:20:41+08:00（Asia/Shanghai）

**两初态均有端到端收益，保留 O1+O2+O3 为当前最佳私有版本。** 同一固定方案的旧 S0 与 O1+O2+O3 的两个完整任务累计耗时 707.183 → 642.318 秒，1.1010×，下降 +9.172%。这是 libero_10/task0 初态3、4各一次的配对 pilot，不能外推所有任务或稳定总体收益。

| 初态 | 旧 S0 总秒 | O1+O2+O3 总秒 | 加速比 | 耗时下降 | 旧/新环境步数 |
|---|---:|---:|---:|---:|---:|
| 3 | 356.615 | 311.441 | 1.1451× | +12.668% | 283 / 283 |
| 4 | 350.568 | 330.877 | 1.0595× | +5.617% | 291 / 291 |

两臂成功分别 2/2、2/2，动作/轨迹长度完全相同；没有通过缩短任务或截断换取耗时下降。

## 耗时分解

| 初态 | 臂 | reset/T5 秒 | 全 policy 请求秒 | 观察/KV请求秒 | 环境执行/客户端余项秒 | outer 总秒 |
|---|---|---:|---:|---:|---:|---:|
| 3 | S0 | 145.215 | 180.393 | 10.743 | 20.265 | 364.851 |
| 3 | O3 | 135.517 | 145.337 | 9.644 | 20.942 | 319.741 |
| 4 | S0 | 135.770 | 182.447 | 10.839 | 21.512 | 358.899 |
| 4 | O3 | 145.209 | 154.495 | 10.597 | 20.576 | 339.156 |

新组合的 policy 请求累计下降 +17.365%、减少 +63.008 秒；reset累计差 +0.258 秒。policy请求包含 video、draft、verify、接受/回退、输出拷贝和 RPC 传输；不把全部请求差额归因为单一 GPU kernel。客户端余项是总耗时减所有模型RPC，含真实环境执行等，未做受扰动 profiler 分摊。

主指标为原 client elapsed_sec：包含reset、prompt/T5、draft、验证、回退、动作提交和环境执行，保留原完整800步horizon。outer额外含环境构造与关闭。原server构造/模型加载、进程准备的Python模板、任务返回后的async保存drain不在task计时内，仍计入GPU预算。O2/O3 helper模板在服务准备阶段生成；O1前缀模板首次生成仍计入首个policy请求。S0每chunk原安装逻辑包含于policy时间；新臂保持新请求绑定与缓存生命周期，热chunk无AST重建。

## 精度、覆盖与运行安排

第一批两独立EGL worker：S0/init3、O3/init4；第二批交换：O3/init3、S0/init4。固定同checkpoint、训练draft A_off、seed20260922、global selector、threshold0.007839680526450805、video20/action50、CFG、VAE设备；profile/full trace/teacher关闭。prompt与cross-KV未共享，rowproof保持原规则。

新增CPU六组BF16完整50步helper对照覆盖首块frame0和后续frame4、全拒绝/全接受/生产阈值；真实tiny生产30层模型，30通道输出、全部KV、scheduler/RNG和接受计数一致。tiny CPU reset夹具硬编码float32 KV，首轮在0 GPU时拒绝；只修正可抛弃夹具的KV dtype，失败日志保留。

独立CPU读取本轮完整原始存档：74对action/video张量、927072元素、574条真实执行命令，以及80对初终环境/controller状态数组逐字节一致；初始观察、环境XML/init、终态与客户端RNG一致，全部target调用分布、接受深度、输出与完整任务步数一致。另外独立CPU读取历史原Torch K2（未共享）完整两局：本轮两臂与其148对action/video、1148条命令及初终环境/客户端RNG一致。历史数据只作为数值oracle；性能仍使用本轮实测S0，无历史耗时替代。

本轮未保存整局每个chunk的完整KV快照，整局结论来自全部输出/命令/环境/RNG一致；此前O3长KV完整chunk全30层有效KV字节/元数据一致的门仍冻结保留。不将整局输出一致描述成整局每个KV均被独立归档检查。

## 资源、退出与下一步

四个完整episode，保留第一批并补齐第二批；共八model启动、三个有限GPU窗口。第二批的两次启动在VAE/checkpoint加载阶段发现外部GPU进程，被原严格guard中止；0 target经constructor/无Runtime/无client证明，CUDA加载时间仍计费。旧FAILED原保守计数与清理失败保持不改，RECONCILED-CLOSED另记0前向/最终退出与稳定空闲。未启动的两任务通过有限续跑补齐；最后增加启动等待门，仅在client启动前等待外部占用消失并连续15秒确认，计数必须0；正式任务继续立即拒绝外部占用。没有重复任何已完成性能样本；计费包括首个窗口直到观察到稳定空闲的保守上界。target intents精确 3938，加载至清理 2069.849 秒；累计 33239/35000、11493.013/18000秒，余 1761 次与 6506.987 秒。旧失败保守消耗继续保留。

四客户端自然成功退出；全部async存档drain和计数闭合之后才按精确PID/start/UID/PGID/argv归属结束本次临时server。每批退出记录保留，终态GPU分配0、compute/graphics为空，外部服务未触碰。public源文件、已有worktree状态、protected state/rules/cycle与所有旧证据保持。

先补 O1+O2+O3 组合在短 KV288／中 KV1152 的完整 chunk 对照，固定同状态与原 Torch/旧 S0 口径；预算按本轮实际闭合计数预留。覆盖通过后再研究完整任务环境步数的控制，随后优先研究reset/T5编码开销，它已占新版整局约44%；不先追加更细碎的动作优化。

- [完整逐项结果](../../../投机采样/results/runtime/p0-optimized-e2e-resume-v2-20261013-160413/FINAL_RESULT.json)；[CPU预检](../../../review/OMISSIONS.md)；[方案](../../../review/OMISSIONS.md)；[GPU清理和资源](../../../投机采样/results/runtime/p0-optimized-e2e-resume-v2-20261013-160413/GPU-COLLECTION.json)。


## 141. 原LingBot-VA baseline与reset/E2E解释、后续双基线对照约定（2026-10-13T17:06:37+08:00）

# 原 LingBot-VA baseline 对照与 reset/E2E 解释

2026-10-13T17:06:37+08:00（Asia/Shanghai）；只读取既有存档并追加 HW。0 新 episode、0 target forward、0 GPU/CUDA。

## 版本与结论

当前最佳私有版本是 direction 向量投机（K2/B2）＋shared self-attention 历史 KV＋O1/O2/O3。上一轮 1.1010× 是相对已含投机＋共享 KV 的 S0 的增量；原始 LingBot-VA 对照为 T（target-only，不投机、不共享 KV）。这两个分母必须分开。

T 保留原 Torch attention/cache、Long checkpoint、BF16/no-FSDP、CPU 文本编码/GPU VAE，不把它描述成不同部署配置的全体官方 baseline。本次 T 来自 08:40 批次，S0/O3 来自 15:25/16:04 批次；任务0、初态3/4、prompt、seed20260922和模型配置对齐，但批次及并行配对不同，T 对比只属历史参考，尚未在本轮新测 T。

| 初态 | 原 T 总秒 | 上一版 S0 总秒 | 当前 O3 总秒 | T→当前降时 | 环境步数 T/S0/O3 |
|---|---:|---:|---:|---:|---|
| 3 | 360.235 | 356.615 | 311.441 | 13.545% | 293/283/283 |
| 4 | 360.579 | 350.568 | 330.877 | 8.237% | 300/291/291 |

两任务累计 T 720.813 → 当前 642.318 秒，1.1222×，耗时下降 10.890%；环境步数 593→574，减少19步/3.204%。各版均2/2成功，只覆盖同一任务两个初态。初态3少10步/3.413%，初态4少9步/3.000%；chunk数19/19→18/19。时间是两条 episode 耗时之和，不是两个并行worker的批次makespan。

上一版 S0 707.183→当前 642.318 秒，1.1010×，下降9.172%；env_steps 574→574。O1/O2/O3与S0完整输出/实际命令/初终环境/RNG一致，轨迹缩短在原T→投机版本时已出现，不能归功于O1/O2/O3。本轮逐字节验证的历史原Torch K2（D、不共享）是数值oracle，不能与target-only T混淆。

## 为什么局部快而整局比例小

先前所谓“单chunk”部分数据实际是 action verifier replay：共享安装在计时外，输入staging/lease/release在计时内，未覆盖完整视频、动作与环境。新的完整chunk测量包含server.infer，但episode reset在计时外、恢复冻结prompt/KV，仍不含真实环境执行。短/中/长replay下降17.49%/23.42%/34.16%来自固定状态/单worker；短档5/6胜出仍为mixed，不能声称所有KV长度或每个完整chunk稳定同幅加速。

共享KV减少多候选物化/复制历史KV，但当前候选QKV/FFN、cross-attention、视频/VAE、动作输出、CPU同步和环境执行仍存在。7节点验证使用CFG batch14，target-only通常batch2；串行target调用减少不等于每次forward成本一样。direction投机并未把原50次动作scheduler更新删掉。O1复用Python attention适配的AST代码模板（不是verify树/每次重新编译GPU kernel）；O2减少CPU读回；O3固定7行缓冲/批量构造。三个优化不改变接受规则或最终动作。

| 两episode累计项 | 上一版S0秒 | 当前O3秒 | 节省秒 |
|---|---:|---:|---:|
| server reset/T5 RPC | 280.985 | 280.727 | 0.258 |
| 完整policy RPC | 362.840 | 299.832 | 63.008 |
| 观察/KV RPC | 21.582 | 20.241 | 1.341 |
| 环境执行/客户端余项 | 41.777 | 41.519 | 0.258 |
| 完整任务总计 | 707.183 | 642.318 | 64.865 |

policy下降17.365%，但它原来仅占整局51.308%；其余阶段不变时，整局算术降时约8.91%，实测9.17%。节省的63.008秒policy时间实际传递为64.865秒总节省；这里是百分比被固定项稀释。当前reset占总计43.705%。

历史T→当前还存在跨批reset波动：reset 233.115→280.727秒，增加47.611秒；非reset部分487.698→361.592秒，下降25.857%，最终完整任务只下降10.890%。这是观测分解，不证明新增算法导致reset变慢；CPU负载/调度是可能解释，唯一原因未测定。扣除reset的比例只作诊断，不能代替完整E2E。

## reset究竟做什么

调用顺序是环境reset→设置固定初态→5步稳定→一次server reset→循环[policy chunk→执行真实动作→观察/KV更新]。客户端reset在while循环之前，每episode一次；普通chunk不会再次清空历史。见[调用位置](../../../投机采样/scripts/p0_libero_multichunk_client.py)。

[server._reset](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)把frame归0/init_latent置空，清Transformer与streaming VAE缓存，创建空历史KV池，准备动作mask/归一化数据，对本任务prompt编码，设置输出目录并empty_cache。它保持已加载模型权重，不是重新加载模型、生成动作或GPU编译。它使新episode不继承上一次场景的时序记忆；首chunk从空历史开始，随后积累/共享本episode有效KV。初始图像VAE编码在[首chunk _infer](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)，不在reset RPC。

[text编码](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)按512 token padding，在CPU T5编码后把embedding移到GPU；视频CFG启用时[encode_prompt](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)会计算正prompt和空负prompt两个embedding。reset明确传入embeds=None，每episode都重算；得到的embedding在本episode chunk间复用，但prompt/cross-attention投影KV的候选共享仍关闭。

“reset基本没变”严格指S0/当前两初态合计280.985→280.727秒，仅差0.258秒；单初态分别有约±9.5秒波动，并非每次reset完全一样，也不适用于历史T。每局135–145秒是整个server reset RPC，不能全归因于清KV或环境reset；T5是源码支持的重点嫌疑，但T5、clear/alloc与RPC各自占比尚未独立测量。模型构造/加载和环境构造另列，不应称作reset。

## 后续固定对照要求

用户原话：“以后对比除了和上一版对比，同时也要和这个baseline对比可以吗？”

每轮保留原T baseline＋上一版＋当轮候选三臂；冻结baseline实现/配置，但同轮重测其时间。不能永久沿用本次历史720.813秒作为分母。保持checkpoint、任务/初态/seed、动作/视频步数、设备/精度、EGL/并行负载相同并平衡顺序；报完整E2E、分项耗时、成功率、环境步数及chunk数，分别给两种加速比。历史数据只能标注参考。

接续原短/中KV完整chunk计划时加入这两个固定比较口径；原T target-only完整chunk与K2 replay原Torch参考应分开。再按同轮三臂重测E2E；reset/T5先拆分计时确认瓶颈，随后再决定缓存prompt embedding等优化，保持数值与任务结果门。轨迹控制仍按原计划留待性能归因和覆盖后。此次仅解释/记录，不启动新GPU或修改实现。

[逐项CPU计算及原始存档hash](../../../review/OMISSIONS.md)；[上轮完整episode报告](OPTIMIZED-E2E-O4-RESULT-20261013.md)。

## 142. 纯投机收益、真实draft结构与reset影响及叶target闲置分析（2026-10-13T17:30:17+08:00）

# 投机收益为何有限、draft结构与reset/其他开销分析

2026-10-13T17:30:17+08:00（Asia/Shanghai）。HW explain只读取证，另走chat追加研究记录。0新完整episode、0target、0GPU；CPU载入实际A_off权重与原helper做36组消费关系检查，CUDA未初始化。公共实现不改。

## 结论与消融分母

纯direction投机D未证明完整任务稳定收益：原Torch/cache保留时，两个任务合计T 720.813→D 728.401秒，增加1.053%。主要新增成本在policy，reset没有增加。共享KV与O1/O2/O3能减少这套投机实现的开销；当前相对原T历史10.890%属于整套组合，不能称为纯投机贡献，也尚无“相同当前共享/工程基线仅开关投机”的同轮消融。

| 历史两episode累计 | 原T target-only | D direction投机/原Torch | S：D＋共享历史KV |
|---|---:|---:|---:|
| 完整任务秒 | 720.813 | 728.401 | 666.360 |
| reset RPC秒 | 233.115 | 230.195 | 221.836 |
| policy RPC秒 | 428.068 | 439.521 | 388.357 |
| 观察/KV RPC秒 | 20.818 | 19.785 | 18.237 |
| 客户端余项秒 | 38.812 | 38.900 | 37.930 |
| 环境步数/chunks | 593/38 | 574/37 | 574/37 |

各版2/2成功，libero_10/task0、初态3/4。T从v1复用已完成存档，D/S在v3完成，各臂双独立EGL worker同卡、分批；同checkpoint/seed/config但不是随机化多任务因果结论。S增加共享之后观察到D→S降8.517%；新O1/O2/O3另轮对现测S0降9.172%。不同批次S0时间波动明显，不能把这些比例连乘。后续固定原baseline＋上一版＋候选同轮计时协议保持，并保留原D消融说明算法贡献。

## 1. target调用少但每次更重

T每个action普通forward为CFG batch2。K2B2展开root＋2个中间节点＋4个叶节点，一次验证CFG batch14；当前bonus关闭，每树实际前进1或2次scheduler更新，并非通常3步。两任务action target forward 1938→1122（-42.1%），但CFG行调用3876→14556，为3.755倍。行数是工作规模，不是精确FLOPs或wall time；批量摊销与并行仍可能有净收益，但不能只看调用次数。

共1026树、3078次draft前向、6626次树scheduler transition，仅提交1850次实际更新；很多分支最终弃用。50次内部去噪更新没有删除。原T38chunks、D37chunks只少一个；T平均完整policy/chunk 11.265秒，D 11.879秒（+5.451%），所以轨迹短3.20%仍不能抵消每chunk贵。policy累计增加11.453秒，reset少2.920秒、cache少1.033秒，整局净增7.587秒。

短/中/长KV共享实验的分母是相同batch14原Torch verifier D，证明省下历史KV展开/复制成本；不是batch14投机比原batch2 target-only都快。共享没有同时消除各候选当前QKV、FFN、cross-attention、video/VAE、CPU打包/同步。旧同状态OFF动作分支：短KV单需求约快31%、长KV单需求慢4.7–11.2%；同卡双需求长KV慢28–30%。这是旧未共享D，不能描述为当前共享O3也同样慢；双worker反映采集负载，不把它充作用户单需求部署结果。

原成本诊断ON的verifier占action约69–84%，draft约2.2–4.7%，只是受观察器影响的线索，不能给当前E2E精确拆秒。当前首边接受74.561%、两边全接受59.942%，draft提交占74.595%；平均每树前进1.7456步。接受第二边与“第一边接受、第二边用target纠正”当前都前进2步，所以提高全接受比例不自动按比例减少验证次数；优先优化成本而非仅追接受率。

## 2. 额外发现：叶节点target预取未消费

[score_k2_b2_tree](../../../投机采样/scripts/p0_batched_verify.py)明确仅用target第0行计算首边误差、第1/2行计算二层误差；第3–6行叶target留给prefetch。全接受执行返回所选叶target作为prefetch，见[execute](../../../投机采样/scripts/p0_batched_verify.py)。[当前在线循环](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)仅在bonus=True时用它第三次target更新；实际mode=verifier_k2_b2、bonus=False时提交leaf_state/第二个draft方向并step_index+=2，下一棵树不消费prefetch。4个叶状态本身仍用于选路径/提交，不能把“无需计算叶target输出”误解为“可删除叶状态或第二层draft”。

使用原build/score/select/execute与实际A_off CPU predictor：3seed×FP32/BF16×depth0/1/2×global/greedy=36组，保持根/中间target、树状态不变，仅改变4个叶target，误差、选择、accepted_depth、提交状态及next_previous_direction完全相同；全接受时只有prefetch改变。确定性小scheduler验证消费关系，不是完整生产KV或target模型等价测试。

这意味着当前14行中8个CFG叶行的target结果未用于在线接受/后续状态，值得设计3逻辑target节点/CFG6行验证；7个draft状态仍可保留。按既有调用账本同样树数的行数算术14556→6348，但**不是已实现，更不代表GPU时间按比例下降**。实际缩batch可能改变BF16内核/数值；现SharedPrefix/rowproof/buffer硬编码7/14，必须同时适配并验证输出、接受路径、KV及完整任务。不能直接删4行当成已安全优化。

另一候选是利用已算好的叶target启用bonus；它改变推进与后续draft输入/轨迹，需要独立质量/成本验证。当前用户只要求解释，本轮不改bonus或验证树。

## 3. 当前draft模型的实际结构

冻结[A_off.pt](../../../review/OMISSIONS.md)，SHA d55f59c3b467034d21bc21895136bb797c03a752d6934f854ac5e0ee7e031e7c；实际权重shape与loader一致，25928可训练参数，FP32，约101KiB参数存储。

输入227维 = 当前7×4×4=112维动作去噪状态＋前一条实际使用的112维direction＋sigma、next_sigma、timestep/1000三标量。使用冻结feature均值/标准差归一化。

`227 → Linear(227,51) → SiLU → Linear(51,51) → SiLU → Linear(51,224)`

最后224拆为center112与delta112，生成两候选残差center+delta/center-delta；反归一化后加上一direction，返回[2,7,4,4]并转回target状态dtype。见[MLP](../../../投机采样/scripts/p0_batched_verify.py)、[predict输入/解码](../../../投机采样/scripts/p0_batched_verify.py)。direction是flow/diffusion内部更新方向，不是直接预测最终可执行机器人动作。

这是小型双候选自回归残差MLP，没有attention层、视觉/text encoder或自己的历史KV；没有直接输入图像、prompt或Transformer隐藏特征，只通过当前动作状态及上一direction间接获得条件信息。首个去噪更新需要原target得到初始direction；每树root调用一次draft，再对两个中间节点各调用一次，共3次draft forward。

它来自离线teacher方向监督。最近minimal continuation在冻结父权重上使用833个独立A_off样本、1000updates/batch512、学习率1e-4、seed20260922，固定最后一更新；未更新target，不是RL或联合训练。此前训练是否接触评估初态未排除，不能据此声明完全未见任务泛化。结构小且无显式世界条件可能限制proposal适应性，但现证据未证明这是主成本/低收益唯一原因。

## 4. 投机与reset以及其他阶段

**无直接新增reset计算路径。** [infer reset分支](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)仍只调用共同_reset并返回。_reset按videoCFG创建原batch2空cache、清时序/streaming VAE状态、mask/归一化、CPU T5编码正/空负prompt及目录/empty_cache；没有build tree、draft forward或batch14 verifier，也不为投机另编码prompt。shared适配器只包action helper/packer，lease在action finally关闭；O1/O2/O3同样包action，没有修改_reset/encode_prompt。10条已存episode的reset请求账本target forward均0。

T/D/S reset合计233.115/230.195/221.836秒；纯投机D并没有变慢。最新S0/O3合计280.985/280.727秒，也相近。此前原T→当前跨批+47.611秒，只证明批次reset变慢，不能归因于投机。CPU T5、线程竞争、两worker启动/请求交叠、CPU调度、GPU分配/同步等都可能间接影响RPC，未独立拆分，不能断言绝无间接影响或指定唯一原因。

| 部分 | 实现影响与观测 |
|---|---|
| action verifier | 直接增加候选target工作、分支scheduler、打包、评分/选择、finite检查和CPU/GPU同步；共享与O1–O3仅抵消其中部分 |
| draft准备 | [lazy loader](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)在首次action helper读取小checkpoint；predict首次把model移到GPU，属于policy/首chunk，不属于reset；norm统计仍每predict执行.to，GPU finite→Python bool产生同步，其精确当前时间未单独计 |
| 共享安装 | 旧每chunk重新构造Python AST约一秒级，发生action入口；O1模板复用已减少，不是reset编译或每步CUDA编译 |
| video/VAE | 原video20步/编码路径仍运行，投机只针对action；动作/观察不同可能间接改变后续输入/轨迹 |
| 观察/KV更新 | 原更新接口、持久CFG2布局保持；历史D比T减少约1.03秒，没有账面增时 |
| 环境/客户端/RPC | 原调用与horizon不变；不同动作影响步数。历史客户端余项只增加0.088秒；policy客户端/服务器账面差只差约0.019秒，不支持“网络耗时抵消收益” |
| 服务启动/加载 | 进程模板准备/模型bootstrap/任务后async drain另记，不计主任务E2E；不能把它们与reset混算 |

最新policy改进17.365%，旧policy仅占整局51.308%；当前reset约43.705%且没优化，所以完整E2E只降9.172%。但纯投机D的问题是policy本身增加，并非有一大块reset吞掉本已存在的policy净收益。这两个问题需要分别处理。

## 建议次序与未证明项

优先做有效target验证节点/叶prefetch的CPU最小语义方案与成本账本，再在原短/中KV完整chunk计划中固定原T＋上一版＋候选同状态验证；需要时保留D消融，分开算法/共享/主机优化。3-row或bonus只是建议，本轮未实现、无新增GPU预算或测量。然后对reset正/负T5、缓存clear/alloc与RPC拆分；prompt embedding缓存应三臂共同使用，不能算作投机专属收益。先降验证成本，再判断更强draft是否值得训练；轨迹控制原计划保留。

未证明：当前组合投机开关的同轮因果贡献、3-row真实BF16 target/KV/完整任务等价和净性能、reset独立T5秒数、跨任务稳定质量与收益。本轮已证明消费关系与实际checkpoint结构，重算已有10条任务账本，不追加GPU。

[CPU消费检查](../../../review/OMISSIONS.md)；[完整复算](../../../review/OMISSIONS.md)；[既有E2E归因](SPECULATIVE-E2E-ATTRIBUTION-20261013.md)。

## 143. A会加速比计时调研与固定状态回放/完整任务协议（2026-10-13T18:41:43+08:00）

报告：/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/SPEEDUP-EVALUATION-LITERATURE-20261013.md

四篇正式 A 会原文支持区分推理/采样效率与任务质量：建议真实状态完整 diffusion/chunk 计时作为常规速度回归，LIBERO 保留成功率、环境步数与包含 reset 的完整任务耗时；NFE 与每轮推进量不能直接充当墙钟加速比。

本轮研究0GPU。bonus 同后端CPU六组通过，原未共享CPU对照数值差异保留；严格资源门通过后单模型GPU oracle另行进行，未预告精度/速度/闭环结果。下一步：完成 bonus 同轮消融，再将真实状态完整推理作为常规速度回归。

## 144. Bonus完整Torch精度和长KV/六条完整E2E对照（2026-10-13T19:23:30+08:00）

报告：/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/BONUS-E2E-RESULT-20261013.md

bonus真实长KV完整chunk相对原T约1.48×、相对OFF约1.16–1.17×；平均每树推进1.7456→2.4696。三臂均2/2成功，完整任务对T少17.36%、对OFF仅少1.76%，初态3对OFF变慢，尚不证明稳定E2E收益。

固定状态两初态各六轮正收益；六个完整episode均成功。对OFF净省10.426秒，其中policy省14.510秒，reset增加7.089秒。轨迹580步：较T593短、较OFF574长，chunks38 vs OFF37。单模型固定状态和并发完整任务口径分开。

下一步：真实短/中KV完整diffusion/chunk三方回放，再扩大多任务质量评估。

## 145. Bonus默认开启及接受深度选路的完整任务质量对照（2026-10-13T20:55:15+08:00）

本轮按用户新优先级，先开启bonus、检查接受与环境轨迹，再考虑加速比。研究preset已更新为bonus ON + depth_then_cost + A_off + 共享KV + O1/O2/O3，原阈值及完整50步不变。

两种bonus选路各完成init3/4两个完整任务，四个任务均成功。GLOBAL为285/295步，DEPTH为285/289步，总环境步580→574；原T历史同态593、旧OFF574。完整接受率69.55%→71.17%，平均推进2.469553→2.497268，但两组仍各38个chunk，因此环境动作少6步不能直接推导E2E耗时收益。

同树“先接受深度、再总误差”选路穷举4096组通过；CPU12对完整50步BF16通过；真实长KV八个完整GPU chunk中，原始Torch/共享O3的全部30层KV、输出、调度器/RNG一致。GLOBAL的112对旧张量、实际动作、初末环境及RNG逐位复现旧BONUS。

高sigma38棵树全部首层拒绝，中sigma72/99首层回退，两区占110/157首层回退；低sigma完整接受87.23%。选路仅两处改变，进一步接受率提升应聚焦真实访问状态与高/中sigma draft，不能盲目增深或放宽阈值。

v1记录器在teacher OFF时访问缺失上下文失败，保存整个失败job/源码，按640次预留和61.885秒保守计入。v2私有请求记录器新增无teacher属性的CPU检查，独立计划通过。离线字段done更正为原schema的done_after_chunk，CPU审计副本独立保存，GPU数据不改。

累计46654次前向、14118.698秒GPU（3.922小时），5小时上限未超。GPU进程全部退出，allocated显存0；5634份冻结输入完整SHA及metadata通过。未宣称本轮新加速比；两状态属于开发pilot。

下一步：先用独立任务/初态补质量门槛；固定阈值与50步，采集bonus真实访问的高/中sigma教师标签改进draft；质量稳定后关闭诊断，测单模型完整policy/chunk与LIBERO完整任务加速比，继续对比原T和旧OFF。

报告：[Bonus接受率与轨迹结果](BONUS-ACCEPTANCE-TRACE-RESULT-20261013.md)。


## 146. Draft结构、训练与质量阈值论文调研（2026-10-13T21:18:39+08:00）

当前draft已联合输出两个同一步候选；旧Transformer对照无稳定优势；阈值来自task8局部误差q90而非任务质量校准。先固定合法前缀与质量协议，再改进真实访问状态训练、比较单层Transformer并独立校准，速度后置。

下一步：独立实现最长合法前缀、同长度只比较保留前缀误差的CPU候选；锁定数据划分与质量协议，再以原MLP和阈值为参照逐项验证训练及结构。

报告：[DRAFT-DESIGN-ACCEPTANCE-LITERATURE-20261013.md](DRAFT-DESIGN-ACCEPTANCE-LITERATURE-20261013.md)

记录：CHAT-DRAFT-DESIGN-LITERATURE-20261013；本轮0 GPU、0训练、0新episode，新方案均待实施。


## 147. 合法前缀选路完整任务质量与标签审计（2026-10-13T22:16:04+08:00）

合法前缀选路细化通过4096组穷举、12对完整BF16 CPU及8个原Torch GPU chunk；六条完整任务全成功。上一版选路DEPTH与本轮前缀误差修正版PREFIX在init6/7均298/258步、总556，原T313/250、总563；同初态输出/动作/终态/RNG完全一致，未观察到新收益。旧标签可用但缺当前bonus训练张量，下一步补教师标签并改进MLP。

下一步：补bonus真实访问状态教师标签，再固定MLP/阈值做训练消融；结构与速度随后。

报告：[PREFIX-ACCEPTANCE-QUALITY-RESULT-20261013.md](PREFIX-ACCEPTANCE-QUALITY-RESULT-20261013.md)

记录：CHAT-PREFIX-ACCEPTANCE-QUALITY-20261013；CPU/GPU完整性通过，无新墙钟加速结论。


## 148. 大规模LIBERO性能调研延期要求（2026-10-13T22:19:37+08:00）

用户要求暂缓LIBERO提速：小规模采样/测试允许等待；未来启动大规模采样或测试前，应调研LIBERO仿真耗时与加速方式，并保持结果一致。当前继续教师标签与draft训练主线。

记录：CHAT-LIBERO-SCALE-PERFORMANCE-DEFERRED-20261013


## 149. 同查询教师采集与MLP训练计划、累计GPU新增5小时授权（2026-10-13T23:03:24+08:00）

用户新增5小时GPU额度，累计硬限10小时；采集器6对完整BF16 CPU无扰动、4项损失/梯度/回读与8项启动预检通过。计划两个任务6条采集episode，预计5000–7000标签，4条训练/2条验证；固定MLP/阈值，比较DATA/STEP/NOGAP，再做原T/当前/候选12条完整质量对照。GPU采集已启动，暂无采集/训练/质量结果。

下一步：先完成原Torch/共享与teacher OFF/ON实际GPU无扰动oracle；再采6条完整episode并核查数据，CPU等预算训练三个MLP候选，独立完整质量筛选。

记录：CHAT-DRAFT-TEACHER-TRAIN-PREPARED-20261013


## 150. 当前 bonus 六条完整任务采集与三组 MLP 训练完成（2026-10-13T23:28:16+08:00）

六条完整 LIBERO episode 全部成功，新采5796条同查询标签（训练3906、验证1890）；原Torch/共享KV及采集器开关8个完整GPU chunk精度通过。新增GPU1336.289秒，额外teacher前向0。三组同预算MLP训练已完成，NOGAP通过局部筛选；尚未执行新模型GPU完整任务质量对照。

下一步：为入围NOGAP执行原Torch精度和T/OLD/NEW共12条独立完整LIBERO任务对照；按预先固定的成功、总/最长轨迹和接受率门槛决定采用。

记录：CHAT-DRAFT-TEACHER-COLLECTION-20261013


## 151. 同查询标签训练完成但未改善当前完整任务质量（2026-10-14T00:24:07+08:00）

新采5796条同查询标签，3906训练/1890整episode留出验证，三组各1000次CPU MLP训练后选NOGAP。原Torch/共享KV精度通过；12条完整任务每臂4/4成功，T1130/OLD1057/NEW1105环境步，NEW相对OLD+4.54%/相对T−2.21%；全树接受率75.268%→74.101%，原采用门槛失败，保留A_off。按用户最新授权继续子代理论文/保留训练/exact判据实验。

下一步：用户要求继续实测保持trace与成功率：三子代理并行查原始论文、CPU root已通过保持训练与exact后继判据；主代理统一GPU，依据CPU证据固定新方案，在新初态11/12等做原baseline/上一版/候选匹配完整任务。init9/10仅作本轮诊断，保持累计GPU10h上限。

记录：CHAT-DRAFT-TEACHER-TRAIN-QUALITY-20261014

## 152. 完整任务轨迹与成功率：承接会话后的实测收口（2026-10-14T12:41:48+08:00）

完成16条四臂及12条周期target完整任务；R_GATE_A4采用门未通过，半阈值质量门通过，PF2质量门通过，OLD第二组对T质量门未通过。当前结论限于开发pilot，速度待独立测量。

| 配置 | 成功 | 惩罚步数总计 | 相对T | 相对OLD | 整树接受率 | 每树推进更新 | target前向 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T | 4/4 | 1034 | +0.00% | -0.10% | — | — | 4876 |
| OLD | 4/4 | 1035 | +0.10% | +0.00% | 73.847% | 2.5541 | 2995 |
| TIGHT | 4/4 | 1055 | +2.03% | +1.93% | 50.624% | 2.1438 | 3356 |
| NEW | 4/4 | 1067 | +3.19% | +3.09% | 74.867% | 2.5743 | 3105 |

| 配置 | 成功 | 惩罚步数总计 | 相对T | 相对OLD | 整树接受率 | 每树推进更新 | target前向 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T | 4/4 | 1111 | +0.00% | -3.39% | — | — | 5320 |
| OLD | 4/4 | 1150 | +3.51% | +0.00% | 66.735% | 2.4221 | 3386 |
| PF2 | 4/4 | 1121 | +0.90% | -2.52% | 67.115% | 2.4323 | 4009 |

OLD在本组也通过与T的质量门。NEW完成任务的质量是否通过、整树接受/推进是否改善是两项独立条件；本轮未同时满足，保持原候选preset。半阈值target前向相对OLD变化+12.05%；其质量控制通过不等于优于OLD或提速。全局preset未自动修改。 OLD在第二组相对T的质量门未通过，总步数变化+3.51%、最差配对变化+20.44%。第一组OLD通过仅覆盖对应四个条件，不推定其它初态也通过。PF2的target调用对OLD+18.40%、对T-24.64%；调用成本及批形状不同，不能据此声称墙钟加速。 当前非性能质量任务可共享GPU；性能测量保留原隔离条件。WAM离线训练不能替代完整闭环trace与成功率验收。

下一步：冻结本次PF2实现为下一阶段质量研究候选，现有全局preset暂不改；先核查LIBERO仿真提速与结果一致性，再扩大预声明配对质量验收。训练侧下一轮增加候选访问状态教师标签和完整chunk终点/动作监督，重新用开发闭环选checkpoint；达标后才隔离测完整E2E。

[完整报告](TRACE-SUCCESS-CONTROL-RESULT-20261014.md)；Knowledge CHAT-20261014-TRACE-SUCCESS-TRAINING-PF2-RESULT。

## 153. 投机接受规则的正式论文依据与阈值溯源（2026-10-14T13:52:04+08:00）

投机接受规则来源复核完成：精确复算历史task8/B1的2400条边q90阈值；新增FlowCast(ICLR2026)及ParaTAA、ParaDiGMS、DPM-Solver正式依据，明确局部缺陷与保质保证边界；未运行新GPU实验。

公式有一步更新偏差解释；历史q90不构成成功率保证。FlowCast的速度MSE与原scheduler残差可作为明确文献基线。文献条件、CPU复算和写作建议见[报告](ACCEPTANCE-CRITERIA-CCFA-20261014.md)；[BibTeX](../../../review/OMISSIONS.md)。

下一步：固定PF2和模型，先对比原步长加权L2、FlowCast速度MSE及原scheduler残差归一化；独立开发集校准、完整chunk与配对闭环验收，扩大仿真前先核查提速一致性，通过后再测E2E。

Knowledge：CHAT-20261014-ACCEPTANCE-CCFA-SOURCES。

## 154. 固定PF2的接受器完整任务对照与CVPR倒排（2026-10-14T16:17:13+08:00）

固定PF2的WL2/MSE/真实scheduler残差比较完成：20完整GPU数值chunk与16完整任务独立复核；T:3/4,steps=1640; PF2_WL2:4/4,steps=1144; PF2_MSE:3/4,steps=1655; PF2_RES:4/4,steps=1207；结果仅为开发pilot；Flash-LIBERO代码审查发现当前客户端_data不兼容，尚无物理/SR等价实测；CVPR2027老师稿定11/01 18:00北京时间。

完整配对结果及CPU/GPU控制见[报告](ACCEPTANCE-COMPARISON-RESULT-20261014.md)；CVPR2027与自研MegaTransformer计划见[倒排](CVPR2027-SCHEDULE.md)。

下一步：PF2_WL2在新初态也超出trace门，先处理质量稳定性，保留原T参照与全部历史结果。依据本轮双参照质量门收口接受规则；10/17冻结候选、10/21质量表、10/26自研MegaTransformer verifier原型正确性与隔离E2E、10/30主证据、11/01 18:00交老师完整稿，内部11/15提交。先profile真实verify及WAM reset RPC瓶颈；Flash先做隔离接口/回放桥接。扩大质量及硬件实验沿用剩余GPU预算，不自动变更preset。

Knowledge：CHAT-20261014-PF2-ACCEPTANCE-COMPARISON-CVPR。
