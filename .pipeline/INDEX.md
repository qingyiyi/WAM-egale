# WAM当前入口

更新2026-10-22T15:31:11.420822+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PROGRESS.md)。固定v15，不全量sync。

- M39：[RMS+RoPE融合](experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)，有限精度通过，整层再快0.39%–0.44%，两机采用；仍慢强guarded约9%–11%。
- [完整阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[优化证据索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
- GPU已清理，累计7480.439/36000秒。保留阶段同步；新投影/epilogue流水未实施。M17非独立TEST，M40旧截止过期，四科学权限false。
