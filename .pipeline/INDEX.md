# WAM当前入口

更新2026-10-22T14:42:31.869291+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)与[Cycle](cycles/wam-research-20261016/PROGRESS.md)，固定v15，不全量sync。

- M39：[按需调制融合](experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)，有限精度通过，整层快约1%，两机采用。保持阶段同步；ready负结果保留。
- [当前阶段公式/tile/同步](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。
- GPU已清理，累计7326.682/36000秒；下一项先讨论Norm1/输入调制。M40旧截止已过、M17非独立TEST、四权限false。
