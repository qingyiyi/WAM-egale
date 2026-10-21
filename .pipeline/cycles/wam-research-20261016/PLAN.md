---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T13:45:15.316595+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 6000局与曝光/首两S/长尾缺口保留，非独立TEST |
| `M39` | 简易单层先对齐 | 同源阶段/整层成本；原语合理后再真实DAG补位 |
| `M40` | RoboTwin迁移 | 固定DEV90三臂完整队列，初态/首两S/成本审计 |

## 当前todo

- [x] 前轮逐元素/Cross/FFN组织控制及专用FFN2；负结果全部保留。
- [x] 本轮一个FFN1尾块候选：9真实签名有限逐位、两进程三KV完整层、选后同前序profile；保留0.19%–0.50%小幅改善。
- [x] 算子worker自然退出/回收；源同步本地optimized，baseline/投机生产不动。
- [x] M40 CURRENT18→监督审计→ON3000→部署CAL完成；不是ON闭环SR。
- [x] DEV90入口/路由/清单核查；首attempt加载前导入失败，0teacher/0episode保留。
- [x] 用户明确允许本次补启；新namespace真实导入链CPU通过，GPU1模型READY并开始完整episode。
- [ ] M40跟进固定90局：自然终态、C/P首两S隔离、成本/失败罚分。基础设施/隔离失败停队，不自动重试。
- [ ] M39再量化GEMM主循环/搬运及阶段等待；按真实依赖设计cross K/V提前补位，不凑108任务，不称已实现ready。
- [x] 阶段结果及首组三臂检查点已整理到HW，正在同步GitHub小证据并汇报。

M39累计1905.463629/36000秒；本轮GPU53.236828秒、CPU准备266.815746秒，0失败/0超时，GPU0释放。本轮截止10/21 14:18:37保留，当前作业已完成。下一GPU作业先在原累计内登记有界范围，不能把旧job期限延长。见[结果](../../experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESULT.md)。
M40补启prior25806.971332/86400秒，包含首attempt5.005915秒；DEV90固定截止10/22 01:26:04.706686，单局≤1800秒，开局预留全局+900秒清理，新增≤3GiB，单GPU1模型worker。清单30初态×3臂，不扩大任务/训练/TEST；见[批准范围](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/RETRY-AUTHORIZATION.md)及PLAN.json。

保护baseline/原tile/LIBERO生产、state/cycle/rules；四权限false。BAAI只写~/cxz，不改驱动/服务/时钟、不杀他人。历史见[前计划](history/ffn1-tail-dev90-20261021/PLAN.md)；旧预算/失败/截止不重置。
