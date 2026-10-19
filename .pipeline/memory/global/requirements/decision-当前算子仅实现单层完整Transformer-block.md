---
kind: decision
name: 当前算子仅实现单层完整Transformer-block
level: constraint
scope: cycle:wam-research-20261016;operator_M32
status: active
updated: 2026-10-19T20:32:42.585935+08:00
supersedes:
  - 本轮M32将完整30层重放纳入当前交付和完成条件的安排
sources:
  - 用户明确：不想做完整30层的大算子，只想做一层的算子
---

# 本轮只实现一层完整Transformer block

当前目标是在一层block内部实现tile化与persistent kernel调度，覆盖self-attention、cross-attention、FFN及其norm/RoPE/调制/残差。实现和主要验收收敛到这一层的B2/B14、短中长KV、输出、中间量、有效KV、RNG/状态与性能对照。不得把当前任务扩成30层的大融合或把30层验证作为本轮完成条件。

停止本轮30层fixture迁移和GPU重放准备；此前未完成/失败/账本保留。最终模型E2E仍在后续本机获准的集成阶段处理，不因这次范围收缩宣称已经通过。

baseline及依赖冻结，只改optimized，远端~/cxz隔离实现；原投机质量门、资源截止与证据规则继续有效。单层范围不意味着当前已实现跨子图融合或完整层单kernel，必须按实际代码说明。

地点条款的后续关系：用户随后明确允许四套LIBERO与算子性能统一在远端测量，见[新决定](decision-允许远端统一测量LIBERO与算子性能.md)。因此上文“最终模型E2E后续本机”仅为历史地点约束，已被替代；单层范围、冻结baseline及质量要求继续有效。
