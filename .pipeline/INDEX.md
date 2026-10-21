# WAM当前入口

更新2026-10-22T02:46:38.498460+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)与[Cycle](cycles/wam-research-20261016/PROGRESS.md)。固定v15，不全量sync。

- M39[夜间单层结果](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)：B14三KV下降8.5%/9.7%/16.8%，仍慢于同工作量guarded约11%–12.6%；9签名＋21变体有限逐位。两机安装最终版本，NCU状态看资源账本；硬截止10/22 09:00。
- M40[跨任务4局](experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)已终态/清理，原01:26截止已过，1800局未启动。
- [算子源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。四权限false、旧失败与保护文件保持。
