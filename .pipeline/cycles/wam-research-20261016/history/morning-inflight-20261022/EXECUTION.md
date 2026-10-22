---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-22T07:27:05.479279+08:00'
plan: PLAN.md
progress: PROGRESS.md
---
# 最近Execution（2026-10-22T02:57:15.434151+08:00）

1. 沿用户持续至10/22 09:00、原M39累计10h授权，完成12候选、08b修复、两进程三KV整层及最终逐阶段；负结果和实施失败保留。[作业/证据索引](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)。
2. 本机/BAAI仅optimized安装最终源0cc1944b/DSOac3f9e46，旧二进制保留，所有部署有真实收据；未修改baseline或投机生产。
3. 本机NCU窗口均已终态；计数及是否有效见本轮RESOURCE报告，不将PARTIAL改成PASS。 M39累计保守计费5472.730/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。 剩余预算见[动态账本](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.json)，不得把等待记为GPU实际计算。
4. M40跨任务4局已终态、owned退出，累计54718.243980秒；01:26硬截止不延长，无后继1800队列。
5. 更新派生短入口和Cycle四文件，前版本存history；不运行全量sync，不覆盖保护文件。GitHub main同步以实际commit回执为准。

[前Execution](history/overnight-before-20261022/EXECUTION.md)。

## Checkpoint 2026-10-22T07:27:05.479279+08:00 / M39

用户同意继续；晨间不重置旧账本。三次本机尝试终态：无worker准入策略停止；旧reference ABI失败；隔离正确reference后4范围NCU完成。新数据触发单因素FA2内联候选，远端guard PID820478/birth722365271，CPU构建中。GPU与精度/计时尚未完成，不能写收益。

## Checkpoint 2026-10-22T07:38:44.498027+08:00 / M39

两项源码对照81调用各通过、三KV两进程无稳定改善，恢复原CUDA。构建链新证据触发[前瞻修订1](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/PLAN-AMENDMENT-1.md)：候选上限2→3，仍原09:00/累计10h/本段5400秒；第三项仅编译LTO。03的CPU配置失败1.666秒保留，03b按nvcc要求使用lto_80，CPU构建中。没有扩大机器人或质量任务。
