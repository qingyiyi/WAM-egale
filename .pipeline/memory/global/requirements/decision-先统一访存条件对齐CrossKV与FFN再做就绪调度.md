---
kind: decision
name: 先统一访存条件对齐CrossKV与FFN再做就绪调度
level: constraint
scope: wam-research-20261016/M39
status: active
updated: '2026-10-21T11:14:16.403015+08:00'
supersedes: []
sources:
  - '用户2026-10-21明确：可以的我同意你的方案；承接三KV汇报中的三步顺序'
---
# 先对齐Cross K/V和FFN，再实现就绪tile补位

用户同意执行：先统一前序访存和缓存条件核对Cross K/V；再固定计算体与tile定位FFN1/2嵌入整层增加的成本；主要差距收敛或给出可追溯原因后，再按真实DAG实现就绪tile补位，首个可提前分支考虑只依赖context的cross K/V。

沿用每子算子合适tile与自然任务数，不要求每阶段凑108。每次改动复核短/中/长KV与数值/有效状态；不以调度收益掩盖原语差距。该确认承接现有先对齐再调度要求，未将缓存或寄存器假设认定为根因。

范围仅当前单层optimized和原M39累计10小时预算；具体作业仍有界登记，旧消耗、失败和截止保留。RoboTwin窗口与科学权限不因本次同意而延长或放行。
