---
kind: decision
name: 继续算子优化并先核对当前瓶颈
level: constraint
scope: cycle:wam-research-20261016;operator_research;M39
status: active
sources:
  - 用户2026-10-20明确同意算子工作继续，并要求说明当前4至5倍差距及弥补办法
  - 用户随后明确要求寻找快速GEMM源码直接嵌入，避免启动多个kernel
supersedes:
  - decision-修复后接续RoboTwin有限迁移.md中当次算子仅讨论、未授权新测量的范围
---
# 继续当前算子研究，先核对真实瓶颈

用户同意继续算子工作，要求解释完整单层的实际速度差距，再据证据推进优化。沿现有M39记录与optimized代码，不改baseline、投机侧、原tile或保护HW；保留旧结果、原预算起点及资源上限。

当前版本与旧版本的profile必须分别标注，完整block、attention和线性层GEMM的计时边界不能混用。不能用旧占比声称当前根因已经确定，也不承诺一定弥补数倍差距。

用户随后明确选择寻找成熟快速GEMM源码直接嵌入现有kernel，避免额外kernel启动。保持整层单persistent kernel，优先复用适用SM80的设备mainloop，不将普通host cuBLAS调用误称为warp内tile函数。具体组件由实现核验选择；必须保持原数值门，不预设成熟源码嵌入后必然等于cuBLAS速度。此选择替代此前cuBLAS边界仍在讨论的状态。
