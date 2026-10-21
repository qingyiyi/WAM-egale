# 最近Execution（2026-10-22T02:46:38.498460+08:00）

1. 沿用户持续至10/22 09:00、原M39累计10h授权，完成12候选、08b修复、两进程三KV整层及最终逐阶段；负结果和实施失败保留。[作业/证据索引](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)。
2. 本机/BAAI仅optimized安装最终源0cc1944b/DSOac3f9e46，旧二进制保留，所有部署有真实收据；未修改baseline或投机生产。
3. 本机最终版NCU新窗口等待自然空闲，最迟08:47:30启动、09:00清理；前窗口外部干扰0kernel，原PARTIAL保留。 M39累计保守计费4850.877/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。 剩余预算见[动态账本](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.json)，不得把等待记为GPU实际计算。
4. M40跨任务4局已终态、owned退出，累计54718.243980秒；01:26硬截止不延长，无后继1800队列。
5. 更新派生短入口和Cycle四文件，前版本存history；不运行全量sync，不覆盖保护文件。GitHub main同步以实际commit回执为准。

[前Execution](history/overnight-before-20261022/EXECUTION.md)。
