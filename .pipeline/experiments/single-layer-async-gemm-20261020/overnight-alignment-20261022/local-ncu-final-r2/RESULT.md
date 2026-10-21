# 最终版本本机NCU

状态：COUNTERS_UNKNOWN_OR_PARTIAL；guard=FAILED_OR_PARTIAL，worker=NOT_PRODUCED。

真实middle/B14/action/layer0，冻结DSOac3f9e46；仅诊断，不用于正式延迟表。先前0kernel外部干扰记录保持PARTIAL。

UNKNOWN：未满足无外部干扰的完整counter采集要求，不用部分数据/旧版本计数替代。完整层性能与精度结果独立成立，本次缺口保留。

HYPOTHESIS：local状态保存/恢复、固定资源配置和执行几何可能共同限制整层；这些counter不能单独确定每项耗时。

RECOMMENDATION：结合三KV逐阶段和资源干预解释剩余差距。没有完整native层counter，不宣称整层流量相对native下降或增加某倍数；不使用NCU时间生成speedup。
