# WAM 当前入口

更新2026-10-21T20:09:29.566093+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39[当前源十GEMM核对](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESULT.md)：Cross/FFN无明显集成惩罚，小投影仍有；下一步固定形状设备函数，再FFN原语和DAG。
- [为什么shared写冲突为0](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/BANK-CONFLICT-EXPLANATION.md)：地址枚举与旧NCU吻合；当前数学源未改，原整层3.12%–3.77%收益和九签名有限门保留。
- 累计2373.809/36000秒，GPU0已释放。RoboTwin 20:07:50完整89/90、GPU1继续，10/22 01:26截止不变。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产/保护文件不动，四权限false，失败与统计边界保留。
