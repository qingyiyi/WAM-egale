# WAM 当前入口

更新：2026-10-21T09:13:44.589629+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION；固定v15，不全量sync。

- 算子：table/GELU vec4整层改善2.7%–3.1%；FFN2独立接近Torch，整层并行无稳定收益已撤回。完整层尚未对齐。
- RoboTwin：parent18齐、CHILD/CONT训练和固定GPU接口CAL齐；CURRENT18/ON/DEV待下一窗口。
- 当前两GPU无计算进程。四权限false，全部失败保留，保护源不改。

[短状态](PROGRESS.compact.md) / [算子证据](experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md) / [远端晨报](experiments/robotwin-pilot-20261020/MORNING-RESULT-20261021.md)。
