# T/A_off现有成本账本与原生阶段匹配：CPU读数

日期：2026-10-11T16:27:39+08:00（Asia/Shanghai）。路线：**投机采样 action-only，非tile**。

## 1. 本步结论与完成范围

**CPU整理与交叉检查PASS；本轮新GPU任务/服务/episode=0，训练/teacher/RL/联训=0。**
只读取已完成6局不扰动job和旧受控T参考。分析脚本在临时目录运行，模型/server/client/成本包装/权重/阈值/cache/bonus及protected HW不改。既有原始记录、历史失败和tile路线保留。

关键结论：**当前A_off并非每种chunk都更便宜：短有效KV仍较快，后段长KV明显变慢。少跑target调用并不等于低成本；14行CFG批量验算、后段接受深度下降与树构造等开销必须一起算。**屏障等待和小模型自身forward不是这份账本的主要额外成本。下面是诊断观测，尚非同输入因果实验或自然E2E结论。

此job的完整闭环T为293/300步、A_off为283/291步，均成功；ON与各自OFF全测合同已经精确复现。A_off此处少19个总环境步，但只从38降到37个总chunk。不能沿用历史“只要解决环境步数，单chunk必然更快”的概括，也不能用两个init证明总体任务非劣。

## 2. 来源、物理阶段和覆盖

- 原始来源：[6局job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)；[最终状态](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/status.json)；[冻结plan](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/plan.json)；[T不扰动gate](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/gate-T_ON.json)；[A_off不扰动gate](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/gate-A_off_ON.json)。原plan中的GPU关卡PENDING是启动前意图，最终gate PASS才是结果；不修改原plan。旧T_OFF来源以plan.reference_dir为准，不复用旧夜间A_off。
- 实际逐请求client/server共150个边界：4 reset、75 policy、71 cache。T_ON38/A_off_ON37个policy。实际CFG/input和有效历史选择后的Q/K/V来自真实sidecar；本轮未新采任何张量。
- **阶段取请求开始时的原生In条件＋当前双指接触**，不按chunk号/L2/未来事件硬标签。双指接触不是稳定抓取，In不是释放完成；原生状态可退回较早阶段，标签也随之退回。
- 以`client/chunk_events.jsonl`实际env_steps_before/after链接原生行。**第一chunk实际执行12步，而非16**；后续按记录链接，不用16×chunk猜开始状态。
- 强匹配：同worker/task/init/frame、物理阶段、实际每行输入shape/dtype和归一化QKV签名、planned demand、观测到的server重叠类别及冻结provenance。CFG倍数和调用数是算法结果，不作为强制相等条件。形状相同不代表输入数值相同。
- 原离线分析器没有物理标签时全unmatched；补入上述原生标签后，**64/75请求匹配，覆盖85.33%，组成32个一对一请求对**。增加输入shape/dtype条件后覆盖不变，未强凑匹配。
- **11个请求全部保留为unmatched**：8个在4个对应点物理阶段不同（worker0 chunk15；worker1 chunk5/10/15）；1个是T worker0 chunk18，对側A_off已结束；2个是worker1 chunk18，T双需求/A_off单需求，重叠类别也不同。[全部未匹配原因](../../../review/OMISSIONS.md)与[逐请求阶段](../../../review/OMISSIONS.md)。
- 32对中**仅2对chunk0的完整已测请求初态指纹相同**。其他对即使阶段、KV、需求匹配，连续state/RNG/RGB和query数值仍不同；server RNG/全hidden KV未保存。32对不是32个独立episode，独立init仅2个。

## 3. 32对同阶段/形状/需求请求的成本分解

均为每请求均值；policy屏障不计入server policy计时。缺timer保留null，计数不伪装耗时。

| 项目 | T | A_off | A_off相对变化 |
|---|---:|---:|---:|
| server policy墙钟（秒） | 11.3455 | 13.4557 | +18.60% |
| action total（秒） | 7.7591 | 9.7706 | +25.93% |
| video total（秒） | 3.5187 | 3.6118 | +2.65% |
| action网络forward计时和（秒） | 7.5867 | 8.1033 | +6.81% |
| action其余耗时（秒，非纯CPU时间） | 0.1723 | 1.6673 | +867.43% |
| action实际transformer调用（含terminal） | 51.0000 | 30.4062 | -40.38% |
| action实际CFG行调用单位（不是FLOPs） | 102.0000 | 393.4375 | +285.72% |
| policy全部transformer调用 | 72.0000 | 51.4062 | -28.60% |
| policy全部CFG行调用单位（不是FLOPs） | 144.0000 | 435.4375 | +202.39% |
| policy屏障等待（秒） | 0.0052 | 0.0072 | +37.33% |

action平均增加**2.0116秒**，其中网络forward计时和增加0.5166秒（约25.7%），其他action范围增加1.4950秒（约74.3%）。这只是计时范围分解，非GPU/CPU/内存带宽因果分类。后段饱和KV的网络增量比其余增量更大，不能拿总体74%套到后段。

- A_off action较快10/32对、较慢22/32对；两worker方向一致：worker0 17对action+25.84%，worker1 15对+26.03%。
- 当前K2+B2为7逻辑候选/14 CFG行。action调用51→30.41虽少40.38%，action CFG行调用102→393.44为**3.86倍**。行数是实际工作单位，不是FLOPs；不同batch的调用耗时非线性，不能直接用倍数预测时长。
- A_off平均树构造`action_tree_build` **0.995秒**，其中小模型`action_draft_model` **0.318秒**。构树**包含**draft及部分scheduler，不能再加一遍。候选pack/scatter约0.117秒，verifier准备约0.103秒，error score约0.118秒，route执行约0.056秒。
- `action_tree_scheduler`横跨build/route/ordinary，属于混合嵌套timer，不能整体作为另一个独立成本相加。减去已知不重叠兄弟范围后的action残差T约0.038秒、A_off约0.273秒；未唯一定位为Python、同步或算子。
- policy屏障平均只5.24→7.20毫秒，差约1.96毫秒；transport−server只9.29→11.04毫秒。它们解释不了action约2秒的增量。这个结论不包含屏障改变同时运行负载的间接影响。
- `obs_encode`只frame0有，32对中每臂仅2次记录。条件均值不能当每chunk均值；分摊**已记录**首帧耗时为T 0.0538秒、A_off 0.0553秒。源码明确首帧条件，缺timer不补0。

依据：[逐对CSV](../../../review/OMISSIONS.md)、[逐对JSON](../../../review/OMISSIONS.md)、[结构化分解](../../../review/OMISSIONS.md)。计时嵌套见[公共server源码](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)约1042–1060、1180–1248行；首帧编码约2183行；仅读源码，未修改。

## 4. 有效KV、接受深度与长段变化

这里KV是sidecar实际K.shape中的action有效token数，**不是预分配pool容量**。从144逐chunk增加144，到chunk14达到2160并饱和。以下按现有KV范围描述，不是根据耗时拟合的在线切换阈值，不据此修改算法。

| KV范围 | 对数 | action T→A_off（秒） | server policy T→A_off（秒） | 每次verifier计时（秒） | 平均tree/verifier数 | depth2/tree | draft提交步占50步 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短KV 144–720 | 10 | 7.411 → 6.660（-10.14%） | 11.013 → 10.266 | 0.186 | 26.40 | 72.0% | 82.0% |
| 增长KV 864–2016 | 16 | 7.798 → 9.733（+24.81%） | 11.321 → 13.350 | 0.279 | 27.00 | 64.6% | 77.5% |
| 饱和KV 2160 | 6 | 8.235 → 15.056（+82.83%） | 11.964 → 19.053 | 0.401 | 31.83 | 34.0% | 54.3% |

- verifier每调用计时由约0.186→0.401秒（约2.16倍）；全深度两步接受的树比例下降，verifier次数增加。两种变化叠加，后段A_off action由8.235变15.056秒。
- 不能把KV、物理阶段、连续query难度及GPU并发唯一分开：它们在闭环中相关。记录的是同步profile范围和server并行区间，不是GPU kernel重叠、HBM带宽或单个attention算子的测量。
- **条件性成本启示，非证明：** 当前K2每树最多提交2步，50更新至少需25树。若暂时假设长KV每调用0.401秒保持不变，25树verifier单项约10.03秒，已大于T整段action约8.23秒，尚未计构树等。这说明只盯提高接受率未必解决当前成本；不能推出任何训练都无效，因为调用时延/调度/输入本身可能改变。用实际组件估算当前长KV每调用回本阈值约0.187秒，也不是训练目标或验收阈值。

[实际QKV摘要](../../../review/OMISSIONS.md)保留全部75个policy的block0代表签名/次数；全block覆盖由原账本审计校验。逐KV统计与分组均值在paired-readout中保留。

## 5. 物理阶段与匹配敏感性

| 当前原生阶段 | 对数 | action T→A_off（秒） | server T→A_off（秒） |
|---|---:|---:|---:|
| soup_before_In/current_double_contact | 9 | 7.674 → 8.496 | 11.154 → 11.972 |
| soup_before_In/no_current_double_contact | 10 | 7.411 → 6.660 | 11.013 → 10.266 |
| tomato_after_soup_In/current_double_contact | 4 | 8.329 → 16.421 | 11.996 → 20.453 |
| tomato_after_soup_In/no_current_double_contact | 9 | 7.977 → 11.546 | 11.617 → 15.374 |

细化为当前左/右/双/无接触＋上一条gripper command，仍32对；剔除任一臂chunk执行中跨阶段的7对，剩25对，action仍+22.45%、server仍+16.26%。仅初态全测相同的2对chunk0，action−14.82%、server−8.45%，只反映短KV首段，不能用于后段因果结论。两条初始化的物理时间线、所有crossing保留在结构化账本，未择优删除。

## 6. 完整边界、单需求尾部与OFF描述性核对

| ON臂 | policy数 | policy server时长和 | reset server时长和 | cache server时长和 | policy模型调用和 | policy CFG行调用和 |
|---|---:|---:|---:|---:|---:|---:|
| T_ON | 38 | 432.555 | 254.632 | 20.439 | 2736 | 5472 |
| A_off_ON | 37 | 501.734 | 275.925 | 19.915 | 1899 | 16110 |

上述均为**两worker区间的累加**，有重叠，不能当GPU总用时或episode E2E；这是全量描述而非32对匹配估计。视频依旧约3.5–3.6秒/request，reset每worker约127–138秒，构成大固定成本；reset/cache没有内部phase profile，不猜其网络/CPU占比。reset主transformer sidecar计数0不代表reset无成本，其他组件不在该计数范围。

A_off worker1最后chunk18是自然单需求尾部：KV2160、action5.611秒/server7.807秒、26 verifier；**不并入双需求匹配**。它比之前双worker chunk17快，但query/接受深度/并发都变了，不能据此证明独占GPU的加速或强行补局。

完整且同输出的OFF描述性计时也显示policy/chunk T约11.385秒、A_off约13.204秒（约+15.98%）。但原T_OFF与新A_off_OFF串行、顺序不平衡且无重复timing；ON/OFF的reset亦有显著波动。这不是profile开销的因果估计，不是clean E2E。不能将同输出不扰动PASS理解成耗时无扰动或所有隐藏状态全等。

## 7. CPU验证与保存

**19项本轮CPU交叉检查全部PASS**：150边界独立原始对账、75policy/32对/11未匹配、JSON/CSV均值及phase/KV权重、当前阶段/第一chunk12步、timer缺失/嵌套、有效KV、精确初态敏感性；阶段/demand/overlap/KV/shape/provenance故障注入、缺字段/重复分层拒绝，CFG倍数/调用数不强匹配。标准库分析未import Torch/WAM/LIBERO/MuJoCo，没有GPU分配或服务。

1532既有文件SHA和9target文件size/mtime核对不变，protected state/rules及不存在的cycle保留原状。初版合成匹配fixture漏了chunk字段，8个测试错误；只补齐临时测试fixture后19项通过，初版错误日志保留，实际读数没有修改。未把19项包装成新的真实GPU关卡，也未重跑历史132项CPU工程测试。

保存：[验证JSON](../../../review/OMISSIONS.md)、[CPU测试日志](../../../review/OMISSIONS.md)、[读取脚本](../../../review/OMISSIONS.md)、[测试脚本](../../../review/OMISSIONS.md)、[输入保护manifest](../../../review/OMISSIONS.md)。HW知识/索引/compact/PROGRESS/log增量保存，旧原始记录与tile条目不覆盖；写后保护验证另存post-write-validation.json。

## 8. 下一步（尚未启动，须逐关推进）

1. **先CPU规划固定负载最小成本对照**：短KV/2160长KV × 单需求/双需求，明确普通batch2与verifier batch14、相同权重/cache/input形状与必要数值/初态合同，计时包含网络＋构树＋pack/score/route，不只比较forward数；对不能等价的候选数值明确拆分，不冒充严格同query。
2. 执行前单独确认所需GPU预算、复用/采集输入范围和通过标准；现有证据是否足够直接跳到自然E2E仍未确定。原8局ABBA成本方案不是本轮授权，先据长KV/并发问题修订，不机械自动再跑8局。
3. 成本关确认后，才考虑减少长KV树宽/验算行数、成本感知选择等**候选方案**，保持冻结target，不能现在改阈值/cache/bonus。任何路线要同时验任务成功、环境步数/长尾和profile-off完整E2E，不用L2、接受率或trace长度单项替代。
4. 冻结target真实闭环困难状态训练、RL/KL或大小模型联训继续后置：接受率上升未必回本；当前先定位成本，不作训练有效性结论。本步结束即提醒用户此下一关，**不自动GPU/训练**。
