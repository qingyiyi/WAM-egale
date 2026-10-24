# C-OVERLAP：不采用

OBSERVED：三KV真实27中间量/输出、状态及672×[32A,32B,32MMA,1commit]计数有限逐位通过；新双输入/元数据与单结果槽无热循环全CTA同步。两进程整层均退化，短/中/长延迟增加0.900%/0.603%/0.722%。当前保留A。

HYPOTHESIS：减少同步不能保证净收益；新的跨任务交接、双输入级数/专职warp等成本可能抵消收益，未作单因素stall分解。没有硬件计数器/运行时重叠时长证据，不把源码可重叠等同已定量证明重叠。

UNKNOWN：硬件stall/动态流量、在线verify及机器人效果。四权限及production_admitted=false。

补充：ffn_handoff_counts为成功完成固定循环后的载荷；不是独立硬件TMA请求计数或重叠时长。
