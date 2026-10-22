# WAM当前入口

更新2026-10-22T18:59:32.671003+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。

M39当前先完成FFN定位，生产R2不改。独立FFN相对原Torch约0.81×；关键为最后生产组、FFN2计算/写回，提交等待很小。NCU与SASS补到函数保存恢复的local访问证据，具体可消除收益未知。自有清理，累计10884.364/36000秒，失败/四科学权限false保留。

[诊断结果](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)、[阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
