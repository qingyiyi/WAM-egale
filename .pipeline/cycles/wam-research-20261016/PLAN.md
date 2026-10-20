---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T11:30:55.620554+08:00'
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
    status: completed
---
# 当前有界计划

计算完成与科学验收分列；原终态/失败见[PROGRESS](PROGRESS.md)及[历史](history/M34-complete-20261020/PLAN.md)。

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
| `M39` | completed | [内部FA2](../../experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)：18同输入attention通过、整层9逐位FAIL；216计时/387block、owned清理。旧M39资源FAILED保留 |

当前直接迭代四项已完成：内部FA2实现/有限核验、误差定位、RoboTwin只读复核、资源清理与证据更新。下一建议优先FFN2/Q/Norm2数值，尚未启动该后继作业；不自动扩大机器人/训练/TEST或第二环境。
