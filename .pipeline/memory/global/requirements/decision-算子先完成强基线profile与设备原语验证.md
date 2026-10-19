---
kind: decision
name: 算子先完成强基线profile与设备原语验证
level: constraint
scope: cycle:wam-research-20261016;operator_stage1_2
status: active
updated: '2026-10-19'
supersedes:
  - NEXT-PLAN第1/2阶段仅为待执行建议的状态
sources:
  - 用户明确同意先完成计划1/2，并要求汇报结果和结合源码讲解
  - .pipeline/experiments/operator-stable-timing-20261019/NEXT-PLAN.md
---

# 先完成强基线和设备原语，再汇报代码

当前先执行已讨论第1/2阶段：固定公平强基线，复用现有真实block输入定位热点，再对一个证据支持的子图准备和验证设备原语。完成后一起汇报全部结果、失败/未知及源码调用链。

仅维护baseline与optimized，baseline冻结，新数学代码只在optimized，共用测量工具可完善；投机源码与既定远端评估不改。本次同意不自动扩展到后续persistent scheduler、独立TEST或正式投机性能；实际job先有限登记并遵守自然空闲、原精度和owned清理要求。旧阶段预算和FAILED不重置。
