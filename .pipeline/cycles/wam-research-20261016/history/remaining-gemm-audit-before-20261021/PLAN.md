---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T18:00:18.327657+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 6000局及曝光/初态/长尾缺口保留，非独立TEST |
| `M39` | 简易单层先对齐 | 剩余原语与整层资源差距明确，再真实DAG填尾部 |
| `M40` | RoboTwin迁移 | 固定DEV90自然终态、初态/首两S与成本审计 |

- [x] 旧三KV阶段/CTA/NCU及两8warp整层退化保留。
- [x] 同计算体普通672/108静态/108原子三调度有限对照，差≤0.17%，不再以CTA总数解释慢因。
- [x] 当前8warp主循环/epilogue profile；padding136＋四元素两候选的三KV两进程同前序对照完成。
- [x] 选用context-only改动接入4warp单层；9签名严格输出/中间量/有效KV/RNG门、两进程完整层比较通过；失败/清理保留。
- [ ] 当前源十GEMM阶段×三KV×两进程：独立热输入/同原版前序/整层插桩及扰动对照；见[核对登记](../../experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/JOB.json)。
- [ ] 核对后按依赖DAG提前cross K/V，不以每阶段铺满108为目标。
- [ ] M40固定DEV90自动接续、完整终态汇总，当前57/90；无自动重试。
- [ ] M17曝光/首两S/非独立TEST与长尾缺口保留并总结。

M39累计2305.594158/36000秒，本轮171.141460秒含无效对照30.471349秒，GPU0已释放。本轮19:25:53截止与所有旧截止原样保留，旧作业不重启。用户本次已明确批准继续核对，新阶段GPU≤1200秒、CPU准备编译≤960秒、≤2GiB、总累计仍≤36000秒；新截止独立登记。M40 prior25806.971332/86400秒，10/22 01:26:04.706686硬截止，单局≤1800秒、开局留900秒清理、原3GiB cap。

直接optimized迭代，baseline/原tile/投机生产/保护文件不动，四权限false；BAAI仅~/cxz，不改驱动/服务/时钟，不杀他人。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md)，[前计划](history/projection-gemm-core-before-20261021/PLAN.md)。
