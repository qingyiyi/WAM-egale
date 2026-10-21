---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T15:46:22.848347+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 全6000局及曝光/初态/长尾缺口保留，非独立TEST |
| `M39` | 简易单层先对齐 | 原语映射与整层资源归因；合理后再真实DAG填尾部 |
| `M40` | RoboTwin迁移 | 固定DEV90自然终态、初态/首两S与成本审计 |

- [x] 三KV阶段/CTA、本机NCU完成；等待与计算分开，负结果保留。
- [x] 用户要求投影8warp：大tile及固定tile两个有界对照，各9真实签名和两进程三KV；完整层均退化，已恢复原4warp。
- [x] 源码/二进制回退核对、worker清理、阶段证据及HW收口。
- [ ] 当前已授权：同256×128/8warp计算体的普通672CTA、108静态循环、108原子队列对照；先逐位与tile覆盖，再三KV两独立进程同前序测量。
- [ ] M39后续若继续warp方向：同前序Cross K/V阶段计时及MMA/shared/local计数，区分主循环/写回/编译资源；不直接全面扩8warp。
- [ ] M39小投影/RMS/GELU任务粒度及真实DAG补位，依据证据排序，108是worker池而非各阶段固定tile数。
- [ ] M40固定DEV90自动接续与终态收口，失败不自动重试。
- [ ] M17曝光/首两S/非独立TEST与长尾等已知缺口继续保留并总结。

M39累计2073.412951/36000秒；本轮GPU79.587378秒、CPU266.718863秒，两job终态完成，GPU0释放。新GPU作业须原累计内有界登记，不能延长旧deadline。M40预算prior25806.971332/86400秒，10/22 01:26:04.706686硬截止，单局≤1800秒、开局留900秒清理，原3GiB cap不变。

直接在optimized迭代，baseline/原tile/投机生产/保护state/cycle/rules不动，四权限false；BAAI仅~/cxz，不改驱动服务时钟、不杀他人。

[前计划](history/projection-warps-before-20261021/PLAN.md)，[8warp结果](../../experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)。
