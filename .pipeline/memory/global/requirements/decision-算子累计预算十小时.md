---
kind: decision
level: constraint
scope: wam-research-20261016/M39
updated: '2026-10-20T22:52:59.441874+08:00'
supersedes:
  - M39此前八小时累计GPU上限及当前阶段较小GPU额度
sources:
  - 用户明确“gpu预算提升到十小时吧”
---
# 本轮算子累计GPU预算提升至十小时

当前M39算子GPU生命周期累计上限为10小时；此前实际使用、失败和超时继续计入，不重置起点或覆盖旧账本。新阶段登记引用本次预算变更，已过期旧作业不因此自动恢复。

限现有远端GPU0与当前算子实现、阶段测量、简化复验；不要求用满。RoboTwin的独立预算和截止不变。CPU编译时间与GPU生命周期分列，CUDA active时间无可靠计数时标UNKNOWN。
