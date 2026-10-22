# 当前执行：RMS+RoPE融合已收口

当前阶段同步版保留按需调制，新增B14 self Q/K RMS+RoPE融合。有限9签名/21结构逐位与状态通过；完整层短/中/长2.270464/2.434816/2.449152ms→2.261504/2.424576/2.438400ms，降0.395%/0.421%/0.439%，仍慢最强guarded9.32%–10.93%。普通路径少2.625MiB中间张量、336tile、两处grid同步；不是实测HBM/机器人E2E收益。

两机当前source0f79579d、DSO795be814。GPU已清理，四科学权限false，旧paused72不改。投影/attention QK/PV已核对Tensor Core源码与HMMA机器码；利用率未新测。MegaMoE双槽epilogue流水仅作参考，当前没有实现。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)、[优化证据索引](../.pipeline/experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)、[阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[HW](../.pipeline/PROGRESS.compact.md)。
