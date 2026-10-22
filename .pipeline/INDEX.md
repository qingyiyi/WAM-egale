# WAM当前入口

更新2026-10-22T17:44:23.726236+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。

M39 FFN写回融合有限精度通过，融合本身退化0.83%–1.14%；相对原Torch完整层1.1646/1.6893/2.7445×。仍阶段同步，无新warp流水。GPU已清理，累计8948.557/36000秒；旧失败与四科学权限false保留。

[结果](experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)、[阶段表](experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/STAGE-MAP.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
