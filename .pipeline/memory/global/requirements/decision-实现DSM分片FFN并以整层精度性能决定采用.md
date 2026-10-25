---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;M39-DSM-STREAM
status: active
updated: '2026-10-25T12:04:53.085449+08:00'
sources:
  - 用户明确“我同意你说的，你直接完成代码，看看性能可以吗？”
---
# 实现DSM分片FFN并测精度与完整层性能

用户已批准接续当前DSM分片激活、生产者消费者方案的实际实现和性能验证。保留当前A独立版本；先资源/进度和数值正确性，再FFN与短中长KV完整block比较。保留七段BF16舍入、原baseline及投机代码；不能仅凭减少global写读判定采用。

本轮沿现有5090持续授权，实际job有界，原A100和M42截止不变。设计期尚未执行的描述由本次批准替代，具体性能和采用仍待证据。失败与负结果保留，每项结束打印完整阶段表并标明历史/未测。
