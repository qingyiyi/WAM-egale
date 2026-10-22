---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-22T12:46:50.150956+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前计划：按需调制单因素验证

- [x] M39新ready源码、CPU图与编译。
- [x] 获批45分钟内完成9签名/21结构/三KV六臂两进程。
- [x] 一次clock诊断、入口失败保留与清理；性能不晋级。
- [x] 按用户要求恢复旧11阶段全局同步源码，保留ready证据；活动DSO不变。
- [x] 逐阶段公式、tile数与同步表已完成：[STAGE-MAP](../../../算子研究/wam_ops/optimized/STAGE-MAP.md)；区分数学分块和调度任务。
- [x] 用户已选择仅按需调制融合，保留阶段同步。
- [ ] 完成实现、精度/状态、三KV完整层计时和更新阶段表：[本轮清单](../../experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/PLAN.md)。
- [ ] M39与最强uncached guarded仍有约11%–13%差距；尚未完成完全对齐。
- [ ] M17暴露/配对/长尾证据缺口；M40扩大资源另行处理，不启动。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)。累计6895.808/36000秒，新窗口已闭合；M40旧截止不变。baseline/投机/原tile/保护文件不动。
