# D-ACT-RESIDENT：不采用

OBSERVED：M16×K2048 ACT=64KiB实际驻留，无FFN1重算、无FP32全局新中间量。真实三KV中间量/输出、输入/KV/RNG及336输出tile的七段提交均有限逐位通过。两个进程整层均显著慢于A，短中长耗时为A的2.25558/2.14614/2.03320倍。FFN融合区Torch/Own约0.219/0.220/0.219。

结构变动：原120个FFN1任务＋672个FFN2任务变成98个驻留任务，M16、N128、K32；每驻留任务完整16个FFN1 tile后消费24个FFN2 tile。七段BF16顺序/最后bias/残差保持。容量换来了更小M、更多权重分块与串行消费，多个因素同时变化，不能仅由总时间定量归因某一项。旧全局ACT workspace仍分配，不能宣称显存已节省。

UNKNOWN：动态HBM/L2/DSM流量、所有stall分量、在线verify/机器人闭环。保留A和本负结果、原编译失败。任务trace为循环完成载荷，不是硬件请求计数。四权限及production_admitted=false。
