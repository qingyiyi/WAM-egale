# WAM 当前入口

更新2026-10-21T14:40:28.123288+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39三KV阶段/CTA及本机NCU已完成：[完整慢因分析](experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)。剩余完整层慢14%–29%；重点GEMM映射/local状态、小算子任务粒度，再DAG。
- 本机NCU24秒完成、GPU已释放；本轮累计GPU88.362秒，M39累计1993.826/36000秒。失败与不同计时边界保留。
- RoboTwin GPU1固定DEV90继续；16/90快照已有microwave PF2自然失败，T/C未齐不归因；原截止10/22 01:26。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产与保护state/cycle/rules不动，四权限false。
