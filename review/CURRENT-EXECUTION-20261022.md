# 当前执行证据（2026-10-22T08:00:56.024170+08:00）

**三项晨间候选均未晋级，仍用夜间11；单层尚未完全对齐同工作量guarded强baseline。** 源0cc1944b/DSOac3f9e46已两机核对恢复，baseline/投机/原tile不动，四科学权限false。

|B14 KV|夜间开始ms|保留版ms|uncached guarded ms|仍慢|
|---|---:|---:|---:|---:|
|288|2.510|2.296|2.070|10.94%|
|1152|2.718|2.453|2.187|12.17%|
|2160|2.977|2.476|2.201|12.50%|

原两进程轮换Graph、完整单层；不是30层/任务E2E或cached最强引擎比较。本次重复旧版约2.29/2.46/2.48ms。

- [本轮完整报告](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)、[三候选原表](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/THREE-TRIALS.csv)：FA2强制内联与路由专化无稳定收益；全局LTO慢15.56%/16.15%/15.67%，全部撤回。各9签名/81调用有限逐位通过并不等于性能晋级。
- [当前版NCU](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/local-ncu-reference-fixed/RESULT.md)：local路径232.337MiB、active warp12.5%、Tensor47.93%、barrier warp stall40.67%；不是HBM同量或墙钟占比。准入残留NVML利用率已记录，诊断不作正式计时。
- [LTO阶段归因](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/LTO-STAGE-CHANGE.csv)：中KV Cross-K/V嵌入574/570→772/768µs，而独立约565/566µs；FFN1/2反而199→179、138→125µs。不能简单全局开LTO或全怪同步；具体动态spill归因仍UNKNOWN。
- [下一步](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/NEXT-ALIGNMENT-DECISION.md)：限制FFN设备编译边界并保留Cross-K/V快路径，B14专用入口后备；未实现、未保证收益。cached命中时Cross-K/V不能当可领取的填空任务。当前无ready-DAG。
- [终态与资源](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESOURCE-AND-STOP-REPORT.md)：所有owned退出，M39累计6004.497/36000秒，09:00不延长。原2→3前瞻修订、旧ABI/编译失败、窗口停止均保留。
- RoboTwin工程4局收口、原01:26截止已过，无新1800局；LIBERO6000局齐但暴露/首两S/长尾缺口仍在，非独立TEST。

[现用源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[原夜间证据](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)、[claim矩阵](../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/CLAIM-EVIDENCE-MATRIX.md)。原tensor/DSO/大反汇编留服务器，公开包不冒充独立可运行权重分发。review/RESULTS.json历史paused72身份不变。
