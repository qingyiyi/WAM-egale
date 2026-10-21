# WAM 当前入口

更新2026-10-21T17:04:44.011067+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39[同计算体三调度结果](experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md)：普通672/108静态/108原子差≤0.17%，自有投影仍比Torch慢约11%；36组有限逐位通过，原4warp保持。
- 本轮GPU61.040秒含两脚本失败，累计2134.453/36000秒，GPU0释放。下一步主循环/epilogue，不再凭CTA数解释慢因。
- RoboTwin GPU1固定DEV90，17:02快照45/90，15组初态/首两S通过；自动接续，原10/22 01:26截止不变。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产/保护文件不动，四权限false，失败与统计边界保留。
