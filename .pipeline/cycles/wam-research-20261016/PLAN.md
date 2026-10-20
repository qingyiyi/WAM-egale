---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T17:26:40+08:00'
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
  - id: M40
    status: in_progress
---
# 当前有界计划

| ID | 状态 | 结果/剩余 |
|---|---|---|
| M17 | in_progress | 四套6000局计算齐，旧科学审计未放行，无新GPU队列 |
| M31–M38 | completed | 原静态/persistent/基线/profile/同卡工程与负结果保留 |
| M39 | completed | 当前profile＋固定CUTLASS设备mainloop完成；有限精度通过，仍慢2.54–3.02倍；1497/1500调用，停止本轮GPU |
| M40 | in_progress | 固定专家114/114；原Target首六局5成功1失败并清理；TF适配/训练/DEV未实现齐，不在排队 |

本轮沿[原两卡计划](ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)和[当前用户决定](../../memory/global/requirements/decision-继续算子优化并先核对当前瓶颈.md)，原8h/24h和绝对截止不重置。用户要求整层一个persistent kernel，明确复用成熟设备GEMM，未改为host cuBLAS多kernel。baseline/投机/原tile/保护HW保持。

本次[追加预算与输入](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/BUDGET.md)及[终态](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)：profile12＋门63＋计时42，原1380后共1497；首次CPU构建失败及修复记录齐，不重置计时起点。剩余3次不拼补不完整结论。

下一算子阶段：先定位新主循环的阶段、epilogue与local-memory成本，随后B14专用化减资源压力；成熟原语接近baseline后再评估跨阶段ready填尾波。新的GPU阶段需有明确预算，不把剩余3调用当无界许可。新ready未验证，默认关闭。完整1/3/7×cache强baseline矩阵仍缺。

RoboTwin[六局结果](../../experiments/robotwin-pilot-20261020/FIRST6-TARGET-RESULT.md)不等于TF迁移完成；沿原预算/准备截止推进需先完成Q32/CFG/cache/监督接口。当前按用户优先级收口算子，不为占卡擅加实验。

10/22 go/no-go、10/23冻结实现/claim、10/28主表、11/01 18:00老师中文证据包。四项质量权限仍false。旧计划/记录见[本次前checkpoint](history/device-gemm-followup-start-20261020/PLAN.md)，不覆盖FAILED。
