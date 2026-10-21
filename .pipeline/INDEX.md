# WAM 当前入口

更新2026-10-21T21:15:38.031247+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39[小投影固定形状阶段](experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)：完整层三KV改善1.88%–2.08%，九签名有限逐位；Self-Q退化与旧快独立参考保留。下一步Self-Q边界，再FFN/DAG。
- M40[RoboTwin90终态](experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-RESULT.md)：六任务各5初态，T/C/P成功25/24/23，各30局。非独立TEST；不宣称质量保留，不自动扩大。
- 累计M39 2483.116/36000秒，M40 50006.380/86400秒；owned均退出，两GPU已释放，旧截止保留。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产/保护文件不动，四权限false；失败与统计边界保留。
