---
kind: decision
name: 先融合self-QK的RMS与RoPE
level: constraint
scope: wam-research-20261016/M39
status: active
updated: '2026-10-22T15:19:46.178190+08:00'
supersedes:
  - 当前先讨论Norm1与输入调制的执行顺序
sources:
  - 用户明确同意先融合RMS和RoPE，并要求解释CUDA与Tensor Core以及MegaMoE的warp分工
---
# 先融合self Q/K的RMS与RoPE

在当前阶段同步optimized上，先将self Q/K的RMS归约及其RoPE后处理融合。保持完整3072维RMS归约、原BF16舍入和FP64旋转计算，不合并GEMM，不改变cross-attention数学，不恢复ready队列。

本轮先实现B14候选路径，保留B2兼容路径；核验真实短/中/长输入、审计中间值与有效KV/RNG，再比较完整层同工作量性能。采用与否以精度及整层实测为准，不预设收益。原累计十小时、baseline/投机/原tile保护不变；新的有界作业记录实际生命周期，不重置旧截止。

此决定来自本次用户批准，替代下一项先讨论Norm1的顺序，不覆盖已完成调制结果。源码中的warp角色以当前版本为准；MegaMoE硬件与WAM/SM80的差别必须明确。
