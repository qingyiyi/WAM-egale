---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T09:57:08.243204+08:00'
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
  - id: M38
    status: completed
  - id: M39
    status: pending
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
| `M38` | completed | [整层慢点实测](../../experiments/whole-layer-bottleneck-20261020/RESULT.md)：6签名222block/300原语，5.09–11.14倍慢；GEMM主导、静态领取无益，清理完成 |
| `M39` | pending | 依据M38收敛GEMM/attention设备tile流水与数值边界；下一个作业另登记，不自动复用预算 |

用户最新优先M38，先解释并量化当前慢点，不等待数值修复。所有旧终态与门不变；不做新机器人/训练/TEST或30层。
