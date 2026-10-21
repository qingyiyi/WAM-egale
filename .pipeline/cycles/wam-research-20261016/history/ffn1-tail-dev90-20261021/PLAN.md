---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T13:04:10.565362+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| M17 | LIBERO证据收口 | 保留6000局与曝光/首两S/长尾缺口，不追加TEST |
| M39 | 简易单层先对齐 | 先真实原语及整层资源控制，再真实DAG补位 |
| M40 | RoboTwin迁移 | 固定采集/训练/CAL后接三臂DEV90，分区与预算不变 |

## 当前todo

- [x] table/GELU、RMS/RoPE/调制/残差对照与有限集成；旧负结果保留。
- [x] 同前序控制、Cross KV/FFN专用设备函数；前轮完整层改善9.46%–10.87%。
- [x] 当前同源码独立/整层两进程三KV：嵌入差约±1.4%，不拼接旧编译路径。
- [x] 一次专用FFN2分段并行候选：9签名门、两进程三KV完整层和选后profile；保留0.36%–0.65%小幅收益。
- [x] 清理核验、原因/未知分列、HW/GitHub小证据整理。
- [ ] M39下一步：先FFN1主循环/tile/数据搬运与资源差距，原语达到合理水平后再最小DAG补位。
- [x] M40固定CURRENT18→监督审计→ON3000→GPU部署CAL自动完成，原失败保留。
- [ ] M40下一步：审查DEV90三臂轮换、canonical T/SST、初态/RNG/首两S、成本入口，然后接固定完整评估；当前没有排队。

M39本轮GPU生命周期81.559秒、CPU准备396.740秒、1候选；累计1852.227/36000秒。登记GPU≤1800秒/CPU2/存储≤1GiB，14:07:22绝对截止保留；12:56核验本轮owned全退，未实现ready。详见[结果](../../experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md)。
M40原24h累计25801.965秒，13:02核验后台链自然结束，无worker。继续后台的用户偏好保持；不把缺少评估入口写成作业已排队。新具体DEV job需沿原预算/清单登记，并预留完整episode和清理。

保护baseline/原tile/LIBERO生产与state/cycle/rules；四权限false。BAAI所有写入仅~/cxz；不改驱动/服务/时钟，不杀他人。历史todo/预算见[前计划](history/ffn-resource-control-20261021/PLAN.md)，不重置旧起点。

## 2026-10-21T13:14:36.085469+08:00 两线新接续

- [ ] M40：固定DEV90入口/审计与有界dispatcher，CPU审查后依赖满足即在GPU1接续。
- [ ] M39：一次FFN1有依据的分块/主循环控制，GPU0完成有限正确性和三KV整层对照。
- [ ] 阶段结果、资源及负结果更新HW/GitHub并汇报；不等两线全部结束。

沿原M39累计1852.227/36000秒、M40累计25801.965/86400秒接续，不重置。具体新job另记硬截止/计数与空间帽，不自动重试。
