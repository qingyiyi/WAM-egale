# 已提交 draft 路径局部 score：50 局描述性诊断

本次日志能够准确重建**已记录且实际提交的 draft 路径 score**。这不是对原始 score 构造的重新计算，也不是新的质量验收。诊断 PLAN 在读取这 50 局 COUNTER 前固定；实际全量 CPU 计算 1.400 秒，两核、无 torch/GPU/SSH/新 episode/TEST。

全量覆盖 10 task × init25–29，共 50 局 TF、1049 个 policy chunks、18896 棵 K2/B2 树、31893 条实际提交 draft 边。接受深度 0/1/2 分别 2543/813/15540。全部 113376 个候选 score 的 JSON 数字可精确回到 FP32；对每树重新枚举固定最长合法 prefix 与接受边成本优先规则，并核对实际父/子/深度、原阈值、chunk/frame/context、逐局真实提交计数，全部通过。无 draft 的 chunk 数 0。阈值保持 **0.0078396805264508047**。

## 公式与可得性

冻结 `scripts/p0_batched_verify.py` 的 `_weighted_direction_l2` 实际计算 `abs(Δσ) × ||v_draft.float() − v_target.float()||₂`；O3 标量实现保留此算术，只集中搬运六个 FP32 score。score 作用于模型 compact action direction；没有将其解释为物理动作距离。

`prefix_acceptance_v1/selection.py` 仅在真实 `execute_k2_b2_route` 返回后记录 `actual_depth`，且强制等于合法最大 prefix。每树只加 `first_costs[actual_parent]`（depth≥1）以及 `second_costs[actual_parent][actual_child]`（depth=2）。深度0不加任何 draft score，bonus/fallback 的 target 步不重复计入。逐局 `sum(actual_depth)` 已与 `committed_draft_scheduler_update_count` 核对；draft+ordinary-target+bonus-target 总提交均等于50×实际chunk数。

每个 chunk 的 C_sum 是这些已提交边记录 score 的算术总和；C_mean=C_sum/提交边数。原 COUNTER 的 `context.rollout_chunk` 与 policy request 完全一致，`frame_st_id=4×chunk`。固定 σ 分组使用边源 σ：第一条边为记录 `sigma`，第二条边为记录 `next_sigma`。score 已含步长，**不再次乘步长**。没有原 velocity，不能独立重算 score、核验原 target/draft residual 或给出绝对环境轨迹误差界。

## 成功44局与失败6局

以下均先逐 episode 计算指标，再对 episode 求组均值；chunk 汇总另存，未当作1049个独立试验。6个失败局均 `horizon_reached=true`，日志 policy env_steps 都为796；成功局没有达到 horizon。

| 指标（episode组均值） | 成功44局 | 失败6局 |
| --- | ---: | ---: |
| policy chunk 数/局 | 17.0227 | 50.0000 |
| 已提交 draft 边总数（组内） | 22789 | 9104 |
| score 总和/局 | 1.09489405 | 3.16950283 |
| 每 chunk score 总和 | 0.06434021 | 0.06339006 |
| 每条提交边平均 score | 0.00211407 | 0.00208932 |

成功组749 chunks，失败组300 chunks。失败的每chunk平均累计 score 比成功低 1.477%，每提交边平均 score 低 1.170%。每局总和更大与失败局更长同时出现，不能用总和差认定误差累积导致失败。成功组所有chunk score总和P90=0.07010585，失败组=0.06843962；这是有时长和任务构成差异的描述分布。

6个同时有成败的 task，每个都是4成功/1失败：

| task | 成功/失败局数 | 每chunk score均值 成功/失败 | 每提交边均值 成功/失败 |
| --- | ---: | ---: | ---: |
| 1 | 4/1 | 0.06461043/0.06379207 | 0.00212889/0.00210257 |
| 3 | 4/1 | 0.06651376/0.06605619 | 0.00217678/0.00221220 |
| 4 | 4/1 | 0.06296929/0.06201651 | 0.00206179/0.00203067 |
| 5 | 4/1 | 0.06366455/0.06304581 | 0.00207905/0.00207798 |
| 6 | 4/1 | 0.06187262/0.06185240 | 0.00203148/0.00202529 |
| 8 | 4/1 | 0.06414965/0.06357735 | 0.00211133/0.00208724 |

六个task中，失败局每chunk均值都低于其4个成功局的均值；只有task3的失败局每边均值较其成功组高。这不排除局部关键动作的风险，但当前均值结果**不提供“提高/降低一个累计 score 阈值即可修复失败”的证据**，不能据此选择新阈值。

## 预登记 σ 段：只计实际提交边

| 边源σ区间 | 提交边数 | 平均记录score | P90 |
| --- | ---: | ---: | ---: |
| (0.0,0.2] | 27939 | 0.00171234 | 0.00249721 |
| (0.2,0.4] | 3244 | 0.00445264 | 0.00662355 |
| (0.4,0.6] | 710 | 0.00691124 | 0.00765277 |

(0.6,0.8]、(0.8,1.0] 均0条实际接受边；ZERO/OUT_OF_RANGE均0。不能把“高σ没有接受边”解释为“高σ没有执行动作”，这些位置可以执行原target步。成功/失败各组固定σ段分母和完整分布见 RESULT.json；未按结果重新分段。

## 结论与下一步边界

已获得真实路径的可复用逐chunk诊断表，可用于以后预先登记接受器挑战者；**这批日志本身没有建立新的可靠阈值或成功率保证**。尚无原 velocities、状态敏感度、环境误差传播或反事实闭环证据。后续如研究累计预算接受器，应在训练/CAL或新独立DEV上先登记候选，再做完整配对闭环；本次不调τ、不触发训练、也不把已查看的init25–29继续当无偏最终TEST。

失败局：QUALITY_DRAFT-t8-i25, QUALITY_DRAFT-t3-i26, QUALITY_DRAFT-t4-i26, QUALITY_DRAFT-t6-i27, QUALITY_DRAFT-t5-i28, QUALITY_DRAFT-t1-i29。

原始来源：dispatch `COMPLETE-SMALL-TRACE-ARCHIVE.json` PASS，init25–29五个tar内50个COUNTER，`remote-evidence/fixed104/ANALYSIS.json`。只读源证据和各COUNTER成员列表见 RESULT.json。EPISODES.json保留50局指标，CHUNKS.json保留1049个明确绑定的chunk；没有复制原完整日志、修改FAILED或生成质量PASS。
