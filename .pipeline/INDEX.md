# WAM当前入口

更新2026-10-21T23:02:48.466096+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[Cycle四文件](cycles/wam-research-20261016/PROGRESS.md)。固定v15，不全量sync。

- M40[工程优化结果](experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)：两项及修复组合完整同case有限等价；组合本轮−7.33%，跨任务/失败长尾/重复未验。原失败保留，累计52520.882/86400秒、原10/22 01:26截止，owned退出；1800局/160小时未启动/未批准。
- M39[精简结果](experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)；九签名有限逐位/无稳定退化，Self-Q/FFN→DAG待做，累计2538.680/36000秒。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产/保护文件不动，四权限false，旧失败保留。
