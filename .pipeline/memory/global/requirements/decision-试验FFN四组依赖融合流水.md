---
kind: decision
name: 试验FFN四组依赖融合流水
level: constraint
scope: cycle:wam-research-20261016;M39
status: active
updated: '2026-10-22T17:55:44.193684+08:00'
supersedes:
  - 当前只做FFN写回融合且不执行跨GEMM流水的阶段限制（仅FFN局部）
sources:
  - 用户听取FFN1到FFN2四组依赖方案后明确“那你先融合试试”
---

# 试验FFN四组依赖融合流水

用户批准在当前optimized的B14单层中，试验FFN1+GELU与FFN2+残差的四组依赖流水：按两行块和两K7168分段组织，输入就绪的FFN2可提前执行。保留已有GEMM分块、累加顺序、BF16舍入及残差；第一版保留ACT全局缓冲，不恢复全Transformer就绪队列或扩展warp分工范围。

先有限真实输入/状态/RNG验证，再与同输入同缓存的原Torch进行三KV完整单层对照。原累计10小时预算不增；本次新有界窗口独立登记，旧截止、失败与四科学权限不变。采用与否依据实测，不将尝试授权写成已获得收益。baseline、投机侧及原tile不改。
