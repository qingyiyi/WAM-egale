# WAM 当前入口

更新2026-10-21T15:46:22.848347+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39投影8warp两对照完成：[结果](experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)。大tile慢3.16%–4.85%，固定tile慢1.05%–1.33%；9签名有限逐位均通过，已恢复原4warp。GPU0已释放。
- 本轮GPU79.587秒，M39累计2073.413/36000秒。下一步投影内部MMA/shared/local因果测量，再DAG；完整层差不能写成投影阶段差。
- RoboTwin GPU1固定DEV90继续，15:40快照33/90；11组初态/首两S通过，有自然失败。原10/22 01:26截止不变。
- [短状态](PROGRESS.compact.md)，[旧NCU诊断](experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)。baseline/原tile/投机生产与保护state/cycle/rules不动，四权限false。
