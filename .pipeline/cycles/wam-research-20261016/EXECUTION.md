# 最近执行检查点

## 2026-10-21T15:46:22.848347+08:00 投影8warp两对照完成并回退

- 大tile8warp：三KV完整层慢3.16%–4.85%；固定128tile8warp：慢1.05%–1.33%。两独立进程方向一致，两个候选均不选用。没有独立阶段计时，具体慢因仍UNKNOWN。
- 两版各9签名/23中间量与有效KV/输入/RNG有限逐位通过；合计2748次完整层调用，profile仍单kernel。两机源与远端DSO恢复原4warp。
- GPU79.587378秒，M39累计2073.412951/36000秒；0异常/超时，2性能负结果。owned children自然退出/reaped，GPU0空，RoboTwin GPU1继续。
- [结果及原件](../../experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)、[资源闭合](../../experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESOURCE-AND-STOP-REPORT.json)。
- M40 15:40为33/90，11组初态/首两S通过，microwave valid14三臂失败，basket valid15仅P失败，队列继续；原截止不变。

[前执行](history/projection-warps-before-20261021/EXECUTION.md)。

## 2026-10-21T16:03:04.038104+08:00 原版launch只读核查

- 读取原版B2/B14各三KV trace，共60个GEMM记录，导出grid/block/warp/tile解释；与旧NCU三个B14阶段交叉核验一致。0新增GPU时间，算子源码未改。
- [完整说明](../../experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/TORCH-WARP-ANALYSIS.md)。固定/动态warp是设计建议，不写成用户已决定。
