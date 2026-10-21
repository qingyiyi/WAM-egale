# 当前执行证据（2026-10-22T02:57:22.081650+08:00）

**本轮单层有改善，但尚未完全对齐最强B14同工作量基线。** 最终源0cc1944b、DSOac3f9e46已在两台机器optimized安装。原paused72及四项false保留；安装研究版本不等于生产资格。

|B14 KV|本夜开始ms|最终ms|uncached guarded强基线ms|最终仍慢|
|---|---:|---:|---:|---:|
|288|2.510|2.296|2.070|10.94%|
|1152|2.718|2.453|2.187|12.17%|
|2160|2.977|2.476|2.201|12.50%|

同卡/同输入、两独立进程、轮换Graph计时；相对本夜开始延迟下降8.52%/9.75%/16.81%。这是单层，不是30层/机器人E2E；不使用profiler时间。cached projection未纳入本轮矩阵。

- [完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)、[逐进程完整层表](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/WHOLE-LAYER-COMPARISON.csv)、[三KV逐阶段](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/stage-control-final-r2/RESULT.md)、[claim证据矩阵](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/CLAIM-EVIDENCE-MATRIX.md)。
- 9真实签名＋21结构变体有限逐位；原Torch最终输出、23共有中间量、普通/debug、有效KV/输入/RNG通过。08非16尾曾失败，08b修复，原FAILED保留。不是全部输入定理，也没有在线质量放行。
- 改动：静态point/norm分派、padded写回、既有guarded self嵌入、双4warp attention、FFN1 N256K32半区epilogue、去冗余入口屏障。仍108CTA单层，必要出口依赖保留；没有ready-DAG。
- 剩余慢项self-attention/Cross-K RMS/FFN/GELU，独立执行也慢；资源干预说明整层统一资源配置是部分原因。不能全归因grid等待，不能把静态local936B当动态spill字节。
- [资源与终态](../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.md)：CLOSED，当前累计保守计费5472.730/36000秒，含本机等待。09:00原硬截止；NCU若未取得则UNKNOWN。
- [RoboTwin工程4局](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)终态/清理，basket双方成功−6.19%、microwave双方失败−3.74%，完整受用摘要一致；固定顺序单次探索性，非独立质量N。原01:26截止已过，1800局/160GPU小时未启动/未批准。

[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[CUDA单层](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。原始tensor/DSO/详细流留服务器；公开包边界见EXPORT-NOTES-20261022-OVERNIGHT.json。前派生快照commit `85278471c4e752a08bc2893c57911b5130f9a4ce`。
