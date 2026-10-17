# 固定DEV96：双namespace独立汇总

状态：PARTIAL_OR_GAPPED_FIXED_DEV96_NO_SELECTION。
机器人完整终态72/96，必要审计完整50/96。
原namespace的FAILED/PARTIAL保留；新结果只描述原固定DEV96的有界补齐，四个原权限始终false。

|分层|方案|机器人终态/计划|审计齐全|成功|实际步数总和|失败800均值|
|---|---|---:|---:|---:|---:|---:|
|fresh_DEV|T|20/20|20|20|5311|265.55|
|fresh_DEV|TF_CONT|20/20|20|19|5779|289.15|
|fresh_DEV|TF_ON|20/20|0|20|5222|261.1|
|fresh_DEV|MLP_ON|0/20|0|0|0|None|
|known_risk_DEV|T|8/8|8|6|3244|406.5|
|known_risk_DEV|TF_ON|4/8|2|0|3184|800.0|

表中保留全部已完整机器人终态，包含原失败和关闭审计缺口；缺口不转成算法失败，也不替换为重跑成功。

TF_ON_vs_T配对：{'reference_0_candidate_0': 0, 'reference_0_candidate_1': 0, 'reference_1_candidate_0': 0, 'reference_1_candidate_1': 20}；双方成功步数比：{'n': 20, 'total': 19.68693111360716, 'mean': 0.9843465556803578, 'p50': 0.9907446479309083, 'p90': 1.0203256302521009, 'min': 0.9285714285714286, 'max': 1.027027027027027}。

TF_ON_vs_MLP_ON配对：{'reference_0_candidate_0': 0, 'reference_0_candidate_1': 0, 'reference_1_candidate_0': 0, 'reference_1_candidate_1': 0}；双方成功步数比：{'n': 0, 'total': 0, 'mean': None, 'p50': None, 'p90': None, 'min': None, 'max': None}。

TF_CONT_vs_T配对：{'reference_0_candidate_0': 0, 'reference_0_candidate_1': 0, 'reference_1_candidate_0': 1, 'reference_1_candidate_1': 19}；双方成功步数比：{'n': 19, 'total': 19.057727187235734, 'mean': 1.003038273012407, 'p50': 1.0079575596816976, 'p90': 1.0514162162162162, 'min': 0.9081967213114754, 'max': 1.1101321585903083}。

TF_ON_vs_TF_CONT配对：{'reference_0_candidate_0': 0, 'reference_0_candidate_1': 1, 'reference_1_candidate_0': 0, 'reference_1_candidate_1': 19}；双方成功步数比：{'n': 19, 'total': 18.666149359973993, 'mean': 0.9824289136828417, 'p50': 0.9795081967213115, 'p90': 1.0238155136268343, 'min': 0.8968253968253969, 'max': 1.068592057761733}。

known-risk8：{'reference_0_candidate_0': 1, 'reference_0_candidate_1': 0, 'reference_1_candidate_0': 3, 'reference_1_candidate_1': 0}。
新工程数值门：{'TF_ON_fresh_success_at_least_T': 'UNKNOWN', 'TF_ON_fresh_success_at_least_matched_MLP': 'UNKNOWN', 'nonempty_both_success': 'UNKNOWN', 'length_P50': 'UNKNOWN', 'length_P90': 'UNKNOWN', 'length_max': 'UNKNOWN', 'known_risk_no_new_TF_failure': 'UNKNOWN'}。
物理/RNG：UNKNOWN；严格相机：UNKNOWN。
建议：NO_SELECTION_INCOMPLETE_OR_GAPPED_AUDIT。
