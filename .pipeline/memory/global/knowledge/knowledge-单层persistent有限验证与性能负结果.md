---
kind: knowledge
name: 单层persistent有限验证与性能负结果
level: reference
scope: cycle:wam-research-20261016;M32
status: active
updated: '2026-10-19T21:08:07.097619+08:00'
sources:
  - .pipeline/experiments/persistent-tile-remote-20261019/RESULT.md
---
# 单层persistent已运行但没有新增加速

阶段内部常驻CTA已在self/cross attention和可选FFN1实现；9真实block/90有效状态/48原语有限逐位通过。QKV/Wo/FFN2仍库调用，无跨阶段ready或warp专职分工。

两独立进程B14比同开FFN1静态路径慢4.42%–6.26%，B2 action慢2.56%–2.80%；A/A最大10.07%保留。有限探索不等于完整模型/机器人或最强B14/Graph收益。baseline88项冻结，731block/0full30/0episode，owned全退出。

用户当前只要求一层完整block，停止本轮30层迁移/验证；下一步在这一层内定位常驻开销与真实复用机会，不把常驻本身称为创新或加速。
