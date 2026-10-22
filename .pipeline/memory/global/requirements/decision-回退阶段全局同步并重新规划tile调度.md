---
kind: decision
name: 回退阶段全局同步并重新规划tile调度
level: constraint
scope: wam-research-20261016/M39
status: active
supersedes:
  - decision-当前优先实现依赖就绪tile调度.md中的当前执行优先级
sources:
  - 用户明确要求“你先回退可以吗，我们先回到之前全局同步到阶段”
  - 用户明确要求“我们重新来做顶层规划”
---
# 回到阶段同步版，先重新做顶层规划

用户要求将当前optimized恢复为此前逐阶段全局同步的已验证版本，保留本次ready调度源码、精度/性能和失败证据；随后先讨论顶层设计，不继续直接迭代现有细粒度ready队列。

下一步规划需区分数学tile、调度任务、CTA的数量和职责，讨论更大分块及可合并任务，再决定实施路径。此决定不将任何特定分块大小或融合组合视为已批准结论，不自动启动新的GPU实验。仍限定单层完整Transformer，baseline、投机侧、原tile和保护HW文件不动；旧预算、截止和负结果保留。
