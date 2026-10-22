---
kind: decision
name: 当前优先实现依赖就绪tile调度
level: constraint
scope: wam-research-20261016/M39
status: active
updated: '2026-10-22T12:03:55.494774+08:00'
supersedes:
  - feedback-先逐阶段对齐原版性能再做异步调度.md中的当前先对齐优先级
  - decision-先统一访存条件对齐CrossKV与FFN再做就绪调度.md中的下一步顺序
sources:
  - 用户明确“那要不先这样吧，你看看能不能先去掉同步，实现tile调度”
---
# 当前转向依赖就绪的tile调度

用户已决定当前先实现tile调度，不再以原语完全对齐作为开始调度的前置条件。保留已验证顺序版本及尚慢约11%–13%的历史证据，在现有optimized中替换不必要的全局阶段等待；真实生产者消费者、归约和workspace复用依赖必须保留。

仍限定单层完整Transformer、baseline/投机侧/原tile不动。新调度的数值、状态和完整层性能须重新验证；代码完成不表示收益或精度通过。旧夜间09:00截止和累计十小时账本不改写；此决定记录实现优先级，不将旧作业自动重启。缓存命中时不存在的Cross-K/V计算不得人为添加为填空任务。
