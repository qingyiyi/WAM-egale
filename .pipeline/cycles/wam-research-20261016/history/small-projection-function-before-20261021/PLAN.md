---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T20:09:29.566093+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
| --- | --- | --- |
| `M17` | LIBERO证据收口 | 6000局及曝光/初态/长尾缺口保留，非独立TEST |
| `M39` | 简易单层先对齐 | 解释并缩小原语与整层开销，再真实DAG |
| `M40` | RoboTwin迁移 | 固定DEV90自然终态、配对与成本审计 |

- [x] 同计算体普通672/108静态/108原子对照、旧NCU和8warp整层退化保留。
- [x] context-only padding136与四元素搬运选择；九签名严格有限门和整层收益保留。
- [x] 当前源十GEMM×三KV×两进程，独立/同前序/整层内核对；短KV升频样本保留并补固定预热。
- [x] 地址枚举复现写冲突2064384→0，解释指标及有限范围；owned清理。
- [ ] 当前执行：六小投影固定形状阶段函数，九签名门/三KV两进程完整层/阶段归因；[本轮预算](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/JOB.json)。其次FFN自身差距。
- [ ] 原语合理后才按真实DAG提前cross K/V，不以每阶段铺满108为目标。
- [ ] M40固定DEV90自动接续，当前89/90，完整终态审计；无自动重试。
- [ ] M17曝光/非独立TEST/长尾缺口总结。

M39本轮GPU68.215237秒，累计2373.809395/36000秒，GPU0释放。新21:24:47和原所有截止保留，当前核对阶段已结束。M40 prior25806.971332/86400秒，10/22 01:26:04.706686硬截止，单局≤1800秒/留900秒清理/3GiB cap不变。

仅optimized，baseline/原tile/投机生产/保护文件不动，四权限false；BAAI仅~/cxz，不改驱动/时钟，不杀他人。

[结果](../../experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESULT.md)，[前计划](history/remaining-gemm-audit-before-20261021/PLAN.md)。
