---
kind: decision
name: 调制参数按需融入消费阶段并重测
level: constraint
scope: wam-research-20261016/M39
status: active
updated: '2026-10-22T14:25:12.703928+08:00'
supersedes:
  - decision-回退阶段全局同步并重新规划tile调度.md中的仅讨论不实施本次调制融合
sources:
  - 用户明确同意调制表按需计算，要求融合到消费位置、重新计时并输出完整阶段表
---
# 仅实施调制参数按需计算

在已恢复的阶段同步版optimized中，将temb与固定scale_shift_table的加法移到调制和门控残差消费位置，去掉普通路径的完整MOD物化阶段。保留原数学/精度边界和其余阶段同步，不恢复ready队列，不同时改变tile大小。

先进行真实输入精度与状态核验，再做短/中/长KV整层同轮对照；输出最新完整计算、tile数和同步表。当前授权包含这一有限实现与重测；原累计十小时不重置，不自动复用已过期旧作业，新的有限作业记录实际起止与预算。baseline、投机侧、原tile、保护HW源不动。
