# 四套完整质量结果与原登记门

OBSERVED：6000完整episode，四套各10任务×50初态×T/C/P。表中成功是原环境成功判据，不是脚本exit0或“无FAILED”。所有自然失败保留，实际命令数与失败800罚分分列。原始逐局来源、分任务成功和成本见RESULT.json及EPISODES.csv。

|套件|T成功/500|TF成功/500|PF2成功/500|TF−T百分点|PF2−T百分点|
|---|---:|---:|---:|---:|---:|
|long|483|484|473|+0.2|-2.0|
|goal|463|464|467|+0.2|+0.8|
|spatial|472|466|464|-1.2|-1.6|
|object|497|498|499|+0.2|+0.4|

|套件/方法|T胜/候选败|候选胜/T败|双方成功N|长度P50/P90/max|>1.30局数|
|---|---:|---:|---:|---|---:|
|long/C|11|12|472|1.0000/1.0736/2.3292|6|
|long/P|19|9|464|1.0000/1.0541/3.2236|7|
|goal/C|12|13|451|1.0000/1.0442/2.6667|9|
|goal/P|10|14|453|1.0000/1.0518/4.8889|13|
|spatial/C|11|5|461|1.0000/1.0440/2.7905|18|
|spatial/P|15|7|457|1.0000/1.0330/4.1562|11|
|object/C|1|2|496|1.0000/1.0333/1.4366|3|
|object/P|0|2|497|1.0000/1.0439/2.2938|8|

长度是双方成功子集的补充统计，分位数为type7线性插值；不能删掉失败后只报长度。八个套件/方法组合P50/P90均满足旧工程数值，但max均超过1.30，因此没有一组完整满足原长度门。全体actual steps/失败罚分的均值、总数与逐局值见RESULT.md/json；actual796没有补造为800命令。

|套件/方法|成功数≥T|成功数≥匹配MLP|P50≤1.10|P90≤1.20|max≤1.30|整体门|
|---|---|---|---|---|---|---|
|long/C|True|UNKNOWN|True|True|False|False|
|long/P|False|UNKNOWN|True|True|False|False|
|goal/C|True|UNKNOWN|True|True|False|False|
|goal/P|True|UNKNOWN|True|True|False|False|
|spatial/C|False|UNKNOWN|True|True|False|False|
|spatial/P|False|UNKNOWN|True|True|False|False|
|object/C|True|UNKNOWN|True|True|False|False|
|object/P|True|UNKNOWN|True|True|False|False|

UNKNOWN：这不是独立TEST。数据暴露、选择与首两S严格配对/组件FAILED保留；当前不能据名义配对计数签出独立成功率差区间或3百分点非劣结论。没有全套匹配MLP臂，也没有把历史小DEV的MLP数拼入本表。重复init49的计时轮次不增加独立质量N。

OBSERVED：PF2不是普遍改善。Long比Target少10次成功，Spatial少8次；Goal/Object观察成功数更高，但仍有长度长尾和旧审计限制。连续TF的Long/Goal/Object观察成功数不低于Target，Spatial低6次。不同套的训练身份保留原manifest，不把它们当同一权重跨套泛化。

RECOMMENDATION：当前保留TF64为系统研究结构，权重/接受/调度的质量证书未收口。不要默认打开PF2，不因这些开发结果新增训练或扩大TEST。后续若获准，只针对完整日志里的少数慢成功长尾和首两S数值/状态隔离定位风险，先固定数据隔离与协议，再决定是否值得质量修订。

`engineering_quality_screen=false`、`independent_test_complete=false`、`noninferiority_established=false`、`formal_performance_permission=false`。旧FAILED/retention/NO_GO/paused72/default A_off不改。
