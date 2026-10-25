---
kind: decision
name: 优先验证双CTA混合DSM并分离慢因
level: constraint
scope: cycle:wam-research-20261016;M39本次双CTA
status: active
updated: '2026-10-25T14:55:31.337663+08:00'
sources:
  - 用户明确要求先测试2CTA一组，测完说明到底哪里慢
supersedes:
  - 四CTA负结果后本次不再展开候选的执行优先级；不改旧结果或预算
---
# 优先验证双CTA混合DSM并分离慢因

在既有5090授权内，对当前较大tile/global混合实现补双CTA对照，不能用旧M16完整ACT驻留的E方案替代回答。保留独立A；分开核验cluster启动、DSM交接与主循环改写的成本，先有限数值和状态，再三KV完整层与独立复测，以实测说明可归因与未知项。每作业有界，不修改投机生产、旧负结果、原tile或四科学门。

本次计划：.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/PLAN.md。

## 2026-10-25 用户新授权：FFN1共享B

用户同意在W的1T/4M/3E上实际测试最小2CTA、不同M/同N共享B。旧C2/W负结果不改；本轮不扩大累加器或输出、不改FFN2，L控制分组成本，D测DSM，S仍保留直到完整层实测判定。新预算与输入见`FFN1-CLUSTER2-WEIGHT-20261025/PLAN.md`；不把“觉得会快”写成结果。

## 2026-10-25 用户追加共享A对照

用户要求“如果是共享A而不是共享B你也试试”。本轮scope为FFN1双CTA相同M/不同N共享A，以及同映射独立加载控制；保持数值/warp/tile，先精度后完整层。共享B已闭合负结果不改，新增作业独立预算；不是长期要求必须采用DSM。

## 2026-10-25 迁移准备期间继续算子实验

用户明确要求利用迁移准备时间完成已讨论算子实验并汇报。范围为FFN1共享A交接开销的单一有界候选；FFN2不变，GPU测量与RoboTwin模型worker错开，自有传输在计时中暂停并恢复。结果与采用决定见独立`FFN1-CLUSTER2-GROUP2-20261025`，不从此授权推导新架构搜索。
