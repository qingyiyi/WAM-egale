# 当前讨论（2026-10-22T15:31:11.420822+08:00）

用户批准先融合RMS+RoPE，并追问执行单元、MegaMoE warp角色、投影及attention是否使用Tensor Core、RoPE具体公式。已通过Maintain保存本次实施决定，替代原先Norm1讨论顺序；技术问答保存在本轮证据，不擅自提升为新的调度决定。

B14 self Q/K融合已有限验证并采用，完整层再快0.395%/0.421%/0.439%。原Tensor Core路径保持，当前仍是阶段同步；MegaMoE SM100的双TMEM槽/epilogue独立循环仅作解释，未移植到SM80。RoPE按grid_ids三轴44/42/42维生成旋转角，self Q/K使用，cross/V不使用。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)、[优化索引](../../experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。GPU已清理；下一步继续逐阶段讨论，不自动启动新GPU或机器人任务。
