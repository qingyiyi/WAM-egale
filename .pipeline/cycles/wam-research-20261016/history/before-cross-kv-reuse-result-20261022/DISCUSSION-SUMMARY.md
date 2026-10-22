# 当前讨论（2026-10-22T15:31:11.420822+08:00）

用户批准先融合RMS+RoPE，并追问执行单元、MegaMoE warp角色、投影及attention是否使用Tensor Core、RoPE具体公式。已通过Maintain保存本次实施决定，替代原先Norm1讨论顺序；技术问答保存在本轮证据，不擅自提升为新的调度决定。

B14 self Q/K融合已有限验证并采用，完整层再快0.395%/0.421%/0.439%。原Tensor Core路径保持，当前仍是阶段同步；MegaMoE SM100的双TMEM槽/epilogue独立循环仅作解释，未移植到SM80。RoPE按grid_ids三轴44/42/42维生成旋转角，self Q/K使用，cross/V不使用。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)、[优化索引](../../experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。GPU已清理；下一步继续逐阶段讨论，不自动启动新GPU或机器人任务。

最新用户明确先核查Cross K/V是否只需算一次；已Maintain保存本轮顺序。源码和三档已存context表明每层正/负两组、各组七候选重复；两组不同，三档相同。可复用结论是数学依赖与CPU身份范围，新GEMM布局逐位/性能未测。当前静态baseline已有同形状投影缓存，复用须双方共同比较。

用户追加online RMS调研：当前RMS分支本来就online累加平方和；完整统计量可流式合并，但最终归一化仍依赖整行。PyTorch Multi-CTA post-norm工作硬件为B200，不能照搬DSMEM到A100；FlashNorm预归一化延迟缩放与本项目post-projection RMS不同。论文/浮点约束与UNKNOWN见本轮独立调研，不写成已实现或精度放行。
