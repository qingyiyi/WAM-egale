---
kind: decision
name: 算子后续仅对比原Torch实现
level: constraint
scope: cycle:wam-research-20261016;operator_comparison
status: active
updated: '2026-10-22'
supersedes:
  - 当前算子汇报同时列原Torch与guarded、后续优先对齐guarded的比较安排
sources:
  - 用户明确要求以后只和原torch实现对比
---

# 算子后续仅对比原Torch实现

当前算子后续优化测量和日常汇报以原Torch实现为唯一对照，不自动增加guarded比较臂。保持双方同输入、同缓存条件和计时边界；加速比为T_Torch/T_自有，并保留tile数及每tile工作量。不能将自有native代理或guarded叫作原Torch。

已有guarded结果与原始数据保留，历史报告不重写。改变后续比较范围不改变既有结果、四项科学权限或独立质量要求；仅与原Torch比较的结果只能支持相对原Torch的主张。

本决定替代本Cycle此前的双baseline日常展示与新实验比较安排。随后用户已另行批准FFN写回融合，见[融合决定](decision-先融合FFN写回中的GELU与残差.md)；随后仅FFN局部四组跨GEMM流水已获[新授权](decision-试验FFN四组依赖融合流水.md)；具体结果放实验记录，专职warp重叠仍未作为已实现事实。
