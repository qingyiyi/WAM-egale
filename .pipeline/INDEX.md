# WAM 当前入口

更新2026-10-21T12:25:51.468246+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- 算子：Cross KV/FFN两项完成，固定tile/数学/同步，完整单层再下降9.46%–10.87%，仍慢15%–31%；一个kernel，未ready。
- [本轮直接对照与机制](experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md) / [短状态](PROGRESS.compact.md)。9签名有限逐位通过；动态spill/HBM仍UNKNOWN。
- GPU0本轮已退出；GPU1 CURRENT18采集继续，自动接审计/ON/CAL。DEV90尚未排队，原24h预算保留。
- 每子算子按合适tile，不凑108；下一步先FFN原语/资源差距，再真实DAG补位。四权限false、旧失败、baseline/原tile/投机生产保护不变。
