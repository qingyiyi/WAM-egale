# WAM当前入口

更新2026-10-22T12:46:50.150956+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)和[Cycle](cycles/wam-research-20261016/PROGRESS.md)，固定v15，不全量sync。

- M39：[ready验证与回退](experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)。有限精度通过、性能慢56%–64%；用户要求恢复阶段同步并重新规划，当前源码/DSO旧11。
- [源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[阶段图](../算子研究/wam_ops/optimized/STAGE-MAP.md)。
- GPU窗口已清理闭合，累计6895.808/36000秒。M40旧截止已过，M17非独立TEST，四权限false。
