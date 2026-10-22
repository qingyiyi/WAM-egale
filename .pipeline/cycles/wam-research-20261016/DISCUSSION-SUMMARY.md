# 当前讨论（2026-10-22T17:44:23.726236+08:00）

当前用户要求直接融合FFN并为后续MegaMoE式流水做准备。已完成B14 FFN1+GELU和FFN2+残差3写回融合；GELU/bias/gate/残差仍是普通浮点和特殊函数计算，不能理解成没有CUDA Core相关工作。此版仍阶段同步，MMA与同tile的写回顺序执行，无新专职warp或双缓冲重叠。

9签名/36结构有限逐位及状态通过；相对原Torch完整层短/中/长KV为1.1646/1.6893/2.7445×。同轮融合前为1.1773/1.7085/2.7673×，本次融合反而慢0.83%–1.14%。独立GELU1568任务和残差336任务消失，grid31→29，但数学工作保留；新增epilogue成本尚未被流水隐藏。尾波、指令布局和寄存器影响仅是待验证假设，本版未做逐阶段profile。

后续唯一外部性能对照是原Torch，保持同缓存/输入/边界；旧guarded及所有FAILED仍保留。用户要求的阶段表同时保留tile数、每tile工作量和相对Torch比值；未测的阶段比值标未测，不沿用上一版profile。

建议下一步先分清两处融合增量，再考虑MMA/epilogue双缓冲。当前采用融合版是研究入口选择，不代表性能晋级或生产许可。旧源码/DSO可恢复；baseline、投机与四科学权限不改。GPU作业已闭合，不自动恢复过期机器人作业。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)；[优化索引](../../experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)；[此前讨论](history/before-ffn-epilogue-result-20261022/DISCUSSION-SUMMARY.md)。
