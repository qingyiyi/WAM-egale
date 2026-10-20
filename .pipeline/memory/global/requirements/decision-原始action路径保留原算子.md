---
kind: decision
name: 原始action路径保留原算子
level: constraint
scope: cycle:wam-research-20261016;operator_research;integration_boundary
status: active
sources:
  - 用户2026-10-20询问仅考虑B14的性能，并明确原版模型action部分依旧使用原来的算子
supersedes: []
---
# 原始action路径保留原算子

原始模型的单状态action路径继续使用原来的算子，不因B14 verifier优化而替换。当前性能讨论先单独核算B14路径，原始action不承担自有persistent实现的B2性能退化。

用户的B14问题不等于授权将当前更慢的实现集成到生产，也不自动取消已有B2/video诊断或授权新的GPU性能矩阵。B14仍须独立通过精度、有效状态和性能要求；单层性能不外推完整verifier或端到端收益。
