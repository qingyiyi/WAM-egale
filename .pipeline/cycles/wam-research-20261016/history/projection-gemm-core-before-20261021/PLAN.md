---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T17:04:44.011067+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 全6000局及曝光/初态/长尾缺口保留，非独立TEST |
| `M39` | 简易单层先对齐 | 先GEMM原语与整层资源归因，再真实DAG填尾部 |
| `M40` | RoboTwin迁移 | 固定DEV90自然终态、初态/首两S与成本审计 |

- [x] 三KV阶段/CTA与旧4warp NCU、两8warp候选完整层退化并回退。
- [x] 同256×128计算体普通672/108静态/108原子三调度，精度/覆盖、三KV两进程hot及同前序完成；无有意义差别。
- [x] 整层复核、插桩扰动评估、两脚本失败保留、owned清理和资源闭合。
- [ ] 已启动本轮投影主循环/epilogue核查：本机NCU，远端同输入计时；最多两个依据证据的计算体变体，GPU上限1800秒计入原累计。不能移用旧4warp计数作新根因。
- [ ] FFN/小投影剩余原语与资源差距；合理后按DAG提前cross K/V，不把每阶段凑108当目标。
- [ ] M40固定DEV90自动接续、终态汇总；当前45/90，无自动重试。
- [ ] M17曝光/首两S/非独立TEST与长尾缺口保留并总结。

M39累计2134.452699/36000秒；本轮61.039747秒包含两脚本失败23.611243秒。GPU0释放。新阶段须沿原累计有界登记，旧截止不延长。M40 prior25806.971332/86400秒，10/22 01:26:04.706686硬截止，单局≤1800秒、开局留900秒清理、原3GiB cap。

直接在optimized迭代，baseline/原tile/投机生产/保护文件不动，四权限false；BAAI仅~/cxz，不改驱动服务时钟、不杀他人。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md)，[前计划](history/projection-dispatch-before-20261021/PLAN.md)。
