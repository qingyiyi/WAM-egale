---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T22:14:16.314145+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
| --- | --- | --- |
| `M17` | LIBERO证据收口 | 原6000局/暴露与首两S缺口保留 |
| `M39` | 简易单层先对齐 | 已精简；Self-Q/FFN原语先对齐或解释，再DAG |
| `M40` | RoboTwin迁移 | 官方100次目标；先明确慢因及等价优化，再确认新增资源 |

- [x] 原小投影部分改善，退化/负结果保留。
- [x] Python参数绑定与debug路径精简、移除死函数、9真实签名/普通接口/状态与两进程3KV计时，安装同DSO。
- [x] 代码带读prompt、HW/结果更新。
- [x] 原Target两条完整复现，官方源码/论文口径核查。
- [x] DEV90成本账本、一条完整成功局cProfile、命令一致性与计数语义分列。
- [ ] M39 Self-Q编译边界→FFN原语→ready-DAG。
- [x] M40两单项在完整同case有限等价，相机−4.05%、预热−2.40%。
- [x] 原组合失败保留；用户明确批准修复单局，完整有限等价，255.945秒生命周期，已清理。
- [ ] 两项/组合保留可选工程候选；再核对其他任务/长失败与重复波动，当前不改全局工程默认。
- [ ] 完成慢因改进后更新1800局耗时，新增160GPU小时未获批准，禁止自动扩大。
- [ ] M17原曝光/首两S/长尾缺口。

M39累计2538.679523/36000秒，M40累计52520.881538/86400秒；所有旧deadline保留，当前owned任务均结束。范围为optimized与隔离诊断；baseline/投机生产/原tile/保护文件不动。

[前计划](history/readability-and-runtime-before-20261021/PLAN.md)。
