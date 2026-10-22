---
kind: decision
name: 先融合FFN写回中的GELU与残差
level: constraint
scope: cycle:wam-research-20261016;M39
status: active
updated: '2026-10-22'
supersedes:
  - FFN写回融合仅作建议、尚未执行的状态
sources:
  - 用户在讨论MegaMoE式后续流水后明确说那就融合
---

# 先融合FFN写回中的GELU与残差

用户已批准当前B14单层optimized的FFN1+GELU及FFN2+残差3写回融合。保持原FP32运算和BF16舍入边界；先有限真实输入、有效KV及RNG核验，再以原Torch作为性能对照。沿M39原累计10小时预算，另记本次有界作业，不重置旧截止或失败。

GELU、bias、门控和残差仍有普通浮点/特殊函数计算，融合不是删除这些运算。先减少中间读写和阶段同步；本次不自动恢复跨阶段ready队列或宣称已实现Tensor Core/CUDA Core重叠。baseline、投机侧及原tile不改。
