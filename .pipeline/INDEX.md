# WAM当前入口

先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。动态结果和预算以当前Cycle及对应证据为准。

M39已完成FFN2四元素写回和生产函数内联；完整层继续提速，独立FFN仍慢原Torch约9%。baseline/投机/保护文件不改，四科学权限仍false；M17/M40未恢复过期作业。

[尾部分摊结果](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/RESULT.md)、[最新结果](experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/RESULT.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
