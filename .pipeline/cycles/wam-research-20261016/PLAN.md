---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T09:33:50.273752+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M31
    status: completed
  - id: M32
    status: completed
  - id: M33
    status: completed
  - id: M34
    status: completed
  - id: M35
    status: completed
  - id: M36
    status: completed
  - id: M37
    status: completed
---
# 当前有界计划

M17计算齐但质量审计未放行；M31–M33按原结果完成。M34单层单kernel原型/有限诊断交付完成，原精度与速度NO-GO；completed不代表科学验收通过。M35为已收口，原面板PARTIAL，原360最多330，失败不重跑；M36只读调研完成。逐项证据见[完整PROGRESS](PROGRESS.md)，之前计划保留[归档](history/M34-complete-20261020/PLAN.md)。

M37按用户最新继续授权完成有界验证；M34/M35原终态保留。

| ID | 状态 | 交付 / 验证 |
|---|---|---|
| `M17` | in_progress | 既定6000局计算结束；质量审计未准入，不开新评测 |
| `M31` | completed | 原静态tile结果保留 |
| `M32` | completed | 原阶段内persistent负结果保留 |
| `M33` | completed | 原profile及反事实保留 |
| `M34` | completed | 整层结构交付；原数值/速度NO-GO |
| `M35` | completed | 同卡计时330/360，原PARTIAL不重跑 |
| `M36` | completed | RoboTwin只读调研完成 |
| `M37` | completed | [成本归因](../../experiments/device-contract-20261020/TASK-COST-DIAGNOSIS.md)完成；[唯一设备候选](../../experiments/device-contract-20261020/RESULT.md)四项数值NO-GO，4参考block/12原语/8heuristic，owned退出。异步整层集成未触发 |

下一步M17仅审查既有质量、计时与claim缺口；没有新GPU队列。几何匹配不能复现原数值，不继续盲扫。若后续采用新算术，需单独的数值/闭环重验方案，不能继承旧证书。

新算子研究需先解决Q/FFN2/attention设备数值契约与强cached B14关系；当前不追加GPU网格搜索。10/22若关键证据不足就收缩主张。保护文件、质量四false、范围与资源限制按PROGRESS和约束记忆，不覆盖state/cycle/rules。
