# 登记的六对投入门

决策：NO_GO。只针对当前r3固定panel。

|门|判定|
|---|---|
|six_complete_identity_route|PASS|
|P_all_risk_success|FAIL|
|P_all_control_success|PASS|
|no_lost_success|PASS|
|current_risk_rescued_min1|PASS|
|penalty_risk|PASS|
|penalty_control|FAIL|
|penalty_all|PASS|
|length_p50|PASS|
|length_p90|FAIL|
|length_max|FAIL|
|cost_C_update|UNKNOWN|
|cost_common_success_total_policy|UNKNOWN|

任一已观察共同成功长度比超过1.30即不可消除的FAIL；不能因后续rescued改门。缺失或计时不可比为UNKNOWN。
四项原权限全false，旧TF_ON NO_GO不变；不借历史C算rescued，不授权扩规模/训练，不证明非劣。细节见JSON。
