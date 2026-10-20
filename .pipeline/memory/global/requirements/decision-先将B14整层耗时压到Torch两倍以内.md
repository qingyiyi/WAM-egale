---
kind: decision
name: 先将B14整层耗时压到Torch两倍以内
level: constraint
scope: cycle:wam-research-20261016;operator_research;B14
status: active
sources:
  - 用户2026-10-20要求分析性能差距，先尽可能将耗时差距控制在两倍以内，再进一步调优
supersedes: []
---
# 先达到B14整层两倍以内的阶段目标

用户明确将下一阶段工程目标设为：B14 action的短、中、长KV完整单层耗时，尽可能全部达到同轮原Torch/native的两倍以内，再深入调优。该目标是待实测达到的工作优先级，不是已通过结果、速度保证或最终论文标准，也不替代追平/优于强baseline的长期目标。

保持整层单persistent kernel及现有数值/KV/RNG约束；原始单状态action沿用原算子，baseline/既有投机侧及原tile保护不变。比较同输入、同硬件、无争用、相同cache与计时边界，不能只挑最好KV或用弱baseline达到两倍。

先识别并消除当前融合/适配引入的主要退化。复用核心GEMM源码、一次kernel启动或实现固定warp角色，都不能单独作为性能目标已达成的证明。具体优化路线和瓶颈占比需实验，不把本次临时诊断写成长期根因。

原实验预算、FAILED、四项质量false和独立测试边界继续有效；此分析与阶段目标不自动重置原资源账本或截止。
