---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;experiment:ffn-local-handoff-20261024
status: active
updated: '2026-10-24T17:09:23.828654+08:00'
sources:
  - 用户同意将已讨论FFN方案实际实验，观察是否加速
---
# 有界试验FFN输入、写回重叠及小块驻留

允许在现有5090尝试FFN2输入专职、跨任务MMA与epilogue重叠、小块ACT驻留和双CTA传递。后两者先核实硬件/容量，不把提案当可执行事实。保留当前独立A，原数学/精度标准不变，三KV无插桩整层结果决定采用。逐job有限预算、逐候选完整阶段表和失败保留，基线/投机与A100资源不改。最新持续5090授权适用，旧预算收据保留。
