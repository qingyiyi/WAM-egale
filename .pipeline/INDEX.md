# WAM当前入口

更新2026-10-22T18:16:14.446638+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。

M39四组FFN流水R2有限精度通过；比同轮此前阶段版整层小幅改善0.33%–0.36%，相对原Torch1.1646/1.6808/2.7386×。提前执行已观察，最后组串行链保留。GPU已清理，累计9844.678/36000秒，旧失败与四科学权限false保留。

[结果](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[阶段表](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/STAGE-MAP.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
