---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;M39四CTA验证
status: active
updated: '2026-10-25T13:49:29.799972+08:00'
sources:
  - 用户明确同意先测试4CTA作为cluster，cluster用于减少部分global交互，其他仍可使用原global路径
supersedes:
  - 旧DSM-STREAM结束后不展开下一候选的本次执行限制；不改变旧结果/预算/采用决定
---
# 验证四CTA与global混合复用

用户已批准有限4CTA实验。cluster是局部复用手段，不要求整个FFN都留shared；保留原global路径为基线和必要的跨cluster交接。保留A独立二进制，先数值再FFN与完整层收益，不预设全DSM更优。当前候选不同时引入专职TMA warp，新增设计按独立有界账本执行；不动投机生产入口。

登记：`.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/PLAN.md`。
