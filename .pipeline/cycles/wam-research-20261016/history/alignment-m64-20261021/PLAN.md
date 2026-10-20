---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T23:19:23+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M39
    status: in_progress
  - id: M40
    status: in_progress
---
# 当前有界计划

夜间按[执行安排](OVERNIGHT-PLAN-20261020.md)推进，09:30–10:00收口；新授权不修改旧作业截止。

## M17 — 四套LIBERO证据收口

6000局计算齐，原科学审计未放行；不追加TEST。旧曝光、首两S失败和长尾保留。

## M39 — 简易单层与阶段性能对齐

- [x] 简易版9真实签名、23中间量、18有效KV转移/输入/RNG/重复结果有限逐位验证。
- [x] 同输入native/复杂/简易两独立进程交错比较，简易无稳定退化；活动复杂入口移除，历史归档保留。
- [x] 简易版30阶段×三KV（B14 cache_insert为N/A）的原版/独立/嵌入完整矩阵；662block/4524stage调用计入。
- [x] 主源码中文注释、外置核验、HW/证据说明，旧失败不改。
- [ ] 下一步先pointwise并行组织和B14只读layout，逐项验证；再cross-K RMS、FFN1/cross-attention。
- [ ] M39-A：继续Norm/FFN/cross-attention逐阶段对齐，给出同输入数值与性能和无法对齐的证据理由。
- [ ] M39-B（deferred）：完成M39-A后再讨论跨阶段ready；当前不启动调度试验，保留阶段grid.sync。

[结果](../../experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)、[逐阶段表](../../experiments/single-layer-async-gemm-20261020/stage-alignment/STAGE-RESULT.md)、[资源与止损](../../experiments/single-layer-async-gemm-20261020/stage-alignment/RESOURCE-AND-STOP.md)。当前简易整层比native慢约71%–93%，没有完全速度对齐。用户最新累计GPU10小时，已用/失败计入；本阶段约106.7秒GPU进程生命周期，现GPU0已清理。原240登记保留，新增前瞻修改为720，实际662。旧675/240资源FAILED、1497/1500、372/420不改。

## M40 — RoboTwin固定迁移链

- [x] 首六Target5成功1失败、监督/初态与reset bank核对。
- [ ] 固定剩余42条自动接续；截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。
- [ ] 采集齐后合并Target48 FIT36/CAL12审计，再按原授权固定TF适配/训练/DEV；当前guard不自动训练。

[原登记](../../experiments/robotwin-pilot-20261020/target-remaining42/PLAN.md)仍为21:37:02至次日03:37:02，6h含清理/4GiB/336000teacher，每局1800秒8000teacher，开局预留900秒清理；原24GPU小时帽不变。

baseline/投机/原tile/保护state-cycle-rules不动，BAAI限~/cxz。四质量权限false。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包。历史见[前次PLAN](history/simple-first-start-20261020/PLAN.md)。
