---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-22T07:59:51.237516+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 下一步 |
| --- | --- | --- |
| `M17` | LIBERO证据收口 | 保留6000局、暴露/首两S/长尾缺口；四权限false |
| `M39` | 单层先对齐 | 用户最新决定先实现就绪tile；保留11作已验证对照 |
| `M40` | RoboTwin迁移 | 原截止已过；工程4局已归档，扩大1800局未启动/未批准 |

- [x] 夜间12候选+08b修复、9真实签名+21结构变体、三KV完整层与逐阶段。
- [x] 晨间当前版4范围NCU；旧ABI/窗口失败保留，优化目录依赖校正。
- [x] 2项源码对照及新构建证据触发的第3项LTO，均通过有限精度但无性能晋级；前瞻修订与负结果保留。
- [x] LTO两进程×三KV×28阶段解释退化；活动源/DSO恢复11、两机owned清理。
- [ ] M39完全对齐：仍比同工作量uncached guarded慢约11%–13%，未完成。
- [ ] M39当前：直接在optimized实现依赖就绪tile；[实施清单](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/PLAN.md)。旧FFN编译边界建议延后。
- [ ] ready-DAG：先明确cached/uncached真实任务与原语成本，不把缓存命中时不存在的Cross-K/V当填空任务。
- [ ] M17证据缺口和M40扩大资源另行处理，不自动启动。

[本段结果](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)、[下一步依据](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/NEXT-ALIGNMENT-DECISION.md)、[历史夜间](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)。M39累计6004.497/36000秒、本段531.767/5400秒，原10/22 09:00截止保留；当前没有owned GPU作业。M40累计54718.243980/86400秒，原01:26:04.706686截止已过。范围不变：optimized与隔离证据；baseline、投机生产、原tile、state/cycle/rules不动。BAAI仅~/cxz。

[本段前计划](history/morning-before-20261022/PLAN.md)、[执行中原记录](history/morning-inflight-20261022/PLAN.md)。
