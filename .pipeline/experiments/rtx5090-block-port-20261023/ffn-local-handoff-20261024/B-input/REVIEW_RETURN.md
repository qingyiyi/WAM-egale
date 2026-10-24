# B-INPUT：不采用

OBSERVED：三KV实际27中间量/输出及状态有限逐位通过。只拆TMA输入，两个独立进程整层均退化；短/中/长延迟增加1.201%/1.129%/0.833%。当前保留冻结SELF-QKV-TMA-4MMA。完整阶段表见STAGE-TABLE.md，计时为本轮实测、eager无Graph。

HYPOTHESIS：专职搬运不能自动抵消输入barrier/任务切换成本；没有硬件计数器，不能由整层结果定量归因。C候选继续核验重叠是否能弥补。

UNKNOWN：硬件stall/动态local流量、在线状态及机器人闭环。四权限及production_admitted均false。
