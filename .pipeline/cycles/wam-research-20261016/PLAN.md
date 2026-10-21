---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-22T02:57:15.434151+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

|ID|阶段|下一步|
|---|---|---|
|M17|LIBERO证据收口|6000局的曝光/首两S/长尾缺口保留|
|M39|单层先对齐|12候选终态，11通过有限验证；NCU补证，解释仍慢约11%–12.6%的原因后再决定调度|
|M40|RoboTwin迁移|跨任务4局已终态，原01:26截止已过；1800局和160GPU小时均未启动/未批准|

- [x] 夜间Self-Q、point/norm、写回、attention、FFN与同步对照；前瞻有界修订8→10→12保留，08b是正确性修复。
- [x] 最终9真实签名＋21结构变体有限逐位；B14三KV完整层五臂同轮两独立进程。
- [x] 3KV×28阶段两进程、原版前缀和资源控制；旧FAILED及退化候选保留。
- [x] 最终源0cc1944b、DSOac3f9e46，两机optimized安装同一验证二进制；旧2a8保留。
- [x] 本机最终版NCU有界窗口收口。本机NCU窗口均已终态；计数及是否有效见本轮RESOURCE报告，不将PARTIAL改成PASS。
- [ ] 完全对齐guarded强基线；当前仍有约11%–12.6%差距，不写完成。
- [ ] ready-DAG（尚未实现；先解释/处理原语与统一资源成本）。
- [x] RoboTwin两项＋修复组合、跨任务/长失败4局核验，owned清理。
- [ ] RoboTwin扩大资源/吞吐再评估；旧截止不延长，不自动扩大。

M39累计保守计费5472.730/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。 原累计10h、本轮30000秒/3GiB/09:00不增。M40累计54718.243980/86400秒、原10/22 01:26:04.706686截止已过。四项科学权限false；paused72、A_off、旧retention/FAILED、baseline/原tile/投机生产和state/cycle/rules不动。BAAI仅~/cxz；未新建论文初稿。

[夜间证据](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)、[之前完整计划](history/overnight-before-20261022/PLAN.md)。
