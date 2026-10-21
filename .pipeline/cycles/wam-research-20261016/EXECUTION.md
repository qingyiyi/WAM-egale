# 最近执行

## 2026-10-21T23:02:48.466096+08:00 两项工程优化收口

- 原版/相机/预热三个完整成功case分别243.634/233.767/237.787秒，有限全观察/状态/RNG/命令/RPC审计通过。
- 原组合闭包引用错误0teacher/0动作PARTIAL保留；用户明确授权新namespace单局补跑，修复后225.782秒、全部同样审计通过。相对原版本轮减少7.33%，不是稳定总体或投机E2E结果。
- 原作业781.254秒＋补跑255.945秒；M40累计52520.882/86400，原硬截止不变。两作业owned为空，原固定source未改。
- 两项/组合为可选工程候选；其他任务、长失败及重复计时待做，1800局/160小时未启动/未批准。

[结果](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)、[资源](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESOURCE-AND-STOP-REPORT.json)、[前执行](history/runtime-opt-before-20261021/EXECUTION.md)。
