---
kind: decision
name: 先核查CrossKV按条件分组复用
level: constraint
scope: wam-research-20261016/M39
status: active
sources:
  - 用户明确要求先看第3项，核查K/V是否只需计算一次
supersedes:
  - 上轮助手建议先优化Cross-K RMS内部warp同步的推荐顺序（该建议未成为执行授权）
---
# 先核查 Cross K/V 按条件分组复用

当前先沿真实输入与源码核查 cross-attention K/V 的条件共享范围、CFG差异和已有缓存实现，再决定下一项算子改动。不能把self-attention的候选私有KV视为可共享，也不能把数学上的条件相同直接当作新执行路径已通过逐位验证。

这是本轮算子优化的顺序决定，不预先指定复用实现、不批准改变数值标准、不自动启动新的GPU窗口。用户追加的online RMS问题属于技术调研，其临时推断不提升为长期约束。baseline、投机源码、原tile与保护HW不变；结果沿M39记录。
