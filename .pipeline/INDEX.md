# WAM 当前入口

更新2026-10-21T13:45:15.316595+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- 算子：FFN1尾块有限逐位通过，完整层小幅改善0.19%–0.50%，仍慢14%–29%；嵌入差约±1.3%，下一步原语/等待归因后DAG，未ready。
- [本轮结果](experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESULT.md) / [短状态](PROGRESS.compact.md)。GPU0已释放，M39累计1905.464/36000秒。
- RoboTwin：DEV90获准补启后已在GPU1实际执行，固定队列自动接续；原加载前失败保留，仍于10/22 01:26截止，24h累计不重置。
- baseline/原tile/投机生产保护，四权限false，旧失败/截止不改。每阶段合适tile，不凑108；动态spill/HBM/L2仍UNKNOWN。
