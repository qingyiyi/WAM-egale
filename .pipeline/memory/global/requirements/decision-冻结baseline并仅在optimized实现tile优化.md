---
kind: decision
name: 冻结baseline并仅在optimized实现tile优化
level: constraint
scope: project:operator_code
status: active
updated: '2026-10-19T19:10:25.866447+08:00'
supersedes:
  - decision-完整block基线与优化版均改为原生算子.md中建立M28基线时允许修改baseline的阶段安排
sources:
  - 用户要求baseline不动，optimized从baseline相同实现起步，在optimized中实现tile化Transformer
---

# baseline冻结，优化在optimized推进

用户明确当前完整native baseline保留为固定对照，后续tile化Transformer、persistent worker与调度实现只在optimized中推进。优化版须以与baseline等价的执行路径为起点，再逐项增加改动；未开启优化时应能复现基线行为。仅维护baseline与optimized两份主实现，投机源码、原tile与历史结果不动。

冻结范围包含baseline实际调用的底层源码、编译选项与构建产物，不能仅因共享库位于optimized目录，就通过修改或重建它间接改变baseline。开始优化前应隔离会变化的实现和二进制入口；允许复用只读固定依赖。具体拆分方案及通过状态须核验，不把本决定当成隔离已经完成。

当前已有Q16 FFN1＋GELU融合属于先前局部优化，不能当成新tile/persistent结果。优化版应保留等价基线路径及独立优化开关以便归因，旧结果不改写。每次改动在同输入、状态、计时边界下比较，数值与资源规则继续有效。

这条决定结束M28建立完整原生baseline时的可修改阶段；不否定当时授权与完成结果，也不新增GPU作业或扩大质量权限。

用户随后明确同意先完成执行路径隔离和Transformer tile子图；M31进入实际实现与必要验证阶段。persistent动态调度是后继阶段，本次授权不扩展生产/质量权限。
