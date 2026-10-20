---
kind: decision
name: 继续实现并评估更大tile
level: constraint
scope: cycle:wam-research-20261016;operator_research;M39
status: active
sources:
  - 用户2026-10-20明确要求算子继续实现，tile可以切大一些，并继续关注RoboTwin
supersedes:
  - 上次两倍目标讨论仅做只读分析的当次范围
---
# 继续实现并有限评估更大tile

用户明确要求继续optimized单层算子实现与必要验证，允许评估更大的tile。继续以B14三KV完整层≤同轮Torch两倍为阶段目标，保持精度与公平baseline。较大tile是待比较的实现选择，不要求所有形状统一放大，不预设越大越快。

保持单kernel、原始action沿原算子、baseline及投机/旧tile保护。沿现有M39记录有界接续，不修改旧预算终态或延后原绝对截止。RoboTwin继续关注终态和接续依赖，本条没有明确重启已经过期的机器人链，不把“关注”解释为自动开展训练/DEV。
