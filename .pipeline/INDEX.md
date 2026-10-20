# WAM 当前入口

更新2026-10-20T23:19:23+08:00。先读[策略](CONTEXT-POLICY.md)、[记忆与约束](memory/INDEX.md)，再读[Cycle](cycles/wam-research-20261016/PROGRESS.md)的PLAN/PROGRESS/EXECUTION/DISCUSSION。v15不升级或全量sync。

- [简易版选定/源码与性能](experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)：有限精度通过，无稳定性能退化；当前完整层仍约1.71–1.93倍native。
- [逐阶段矩阵](experiments/single-layer-async-gemm-20261020/stage-alignment/STAGE-RESULT.md)：先pointwise/layout，再cross-K RMS/FFN1/cross-FA2，后ready；阶段barrier仍在。
- [RoboTwin固定42](experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)：截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。GPU1自动继续，03:37:02截止，非TF评测。
- [LIBERO原质量](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：6000局计算齐，科学审计未放行。

[短状态](PROGRESS.compact.md)。四质量权限false、原失败保留；baseline/投机/原tile/保护HW不改，远端限~/cxz。当前算子GPU任务已清理。
