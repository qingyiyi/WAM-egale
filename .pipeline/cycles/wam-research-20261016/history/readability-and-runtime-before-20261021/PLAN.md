---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T21:15:38.031247+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
| --- | --- | --- |
| `M17` | LIBERO证据收口 | 6000局与曝光/初态/长尾缺口保持，非独立TEST |
| `M39` | 简易单层先对齐 | 小投影与FFN原语先对齐/解释，再真实DAG |
| `M40` | RoboTwin迁移 | 固定DEV90完成，审阅质量与样本范围 |

- [x] context写回、同计算体调度、十GEMM同前序、bank/NCU与旧负结果保留。
- [x] 六小投影唯一固定形状候选：九签名门、三KV两进程整层及阶段控制完成，保留净收益和Self-Q退化。
- [x] 同二进制安装、owned清理、结果/HW归档。
- [x] 固定DEV90终态与30配对审计，保留全部失败及旧导入失败。
- [ ] 下一小项：Self-Q专用/通用函数编译边界；保持旧快独立参考，不能移参考制造对齐。
- [ ] FFN自身差距收敛或解释后，再按真实DAG提前cross K/V；不凑108任务。
- [ ] M17曝光/非独立TEST/长尾缺口收口。
- [ ] M40讨论6任务每任务50初态的扩大面板、隔离与预算；尚未启动。

M39本轮109.306358秒、累计2483.115753/36000秒；新22:23:55截止和旧deadline保留，本阶段已清理。M40累计50006.379538/86400秒，原10/22 01:26:04.706686截止不变；当前无运行作业，不自动续轮。

仅optimized、baseline/原tile/投机生产/保护文件不动。四权限false，BAAI仅~/cxz，不改驱动/时钟、不杀他人。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)、[前计划](history/small-projection-function-before-20261021/PLAN.md)。
