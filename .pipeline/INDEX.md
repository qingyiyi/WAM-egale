# WAM当前入口

更新2026-10-22T12:18:20.800737+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)与[Cycle](cycles/wam-research-20261016/PROGRESS.md)。固定v15，不全量sync。

- M39用户转向就绪tile：[源码/CPU结果](experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)。B14依赖图已实现，CPU编译/审查通过；GPU未验，新窗口待确认，旧09:00不重写。默认仍旧11。
- [旧晨间性能/负结果](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)：旧11仍慢同工作量guarded约11%–13%，不作为新调度速度。
- M40原截止已过，M17 LIBERO6000局齐但非独立TEST。四权限false、旧失败/保护文件不变。
- [算子源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[带读Prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。
