# WAM 当前入口

更新2026-10-21T22:14:16.314145+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- M39[入口精简结果](experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)：九真实签名/普通debug有限逐位，整层无稳定退化；[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。Self-Q/FFN→DAG仍待做。
- M40[慢因账本与完整profile](experiments/robotwin-pilot-20261020/runtime-profile-20261021/RESULT.md)：模型reset仅0.0047%，主要为推理/双臂规划/渲染及长尾；[原Target两例复现](experiments/robotwin-pilot-20261020/microwave-reproduction-audit-20261021/RESULT.md)。100次目标保留，新增160GPU小时未批准，1800局未启动。
- M39累计2538.680/36000秒，M40 51483.683/86400秒；owned均退出，原截止保留。
- [短状态](PROGRESS.compact.md)。baseline/原tile/投机生产/保护文件不动，四权限false；旧失败与计数语义保留。
