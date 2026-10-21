---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T11:02:47.052904+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 保留6000局质量/长度/成本与科学缺口，不追加TEST |
| `M39` | 简易单层先对齐 | 真实Torch/剥离原语/整层条件对照，有限精度与性能证据 |
| `M40` | RoboTwin迁移 | 固定Target48→parent18→current18→既定DEV90，预算/分区不变 |

## 当前todo

- [x] 真实Torch GELU向量化/调制入口核验，三KV同输入独立剥离和FFN2写回/分段对照。
- [x] table/GELU vec4入整层：九签名有限逐位、两独立进程完整层改善2.7%–3.1%。
- [x] FFN2并行K独立接近Torch，入整层未受益，保留负结果并恢复较快源/DSO。
- [x] 用户授权补齐parent18，完整监督审计，CHILD/CONT固定训练与GPU部署接口CAL。
- [x] 全部本轮进程退出，原始失败/预算/截止保留，短HW更新。
- [x] M39：官方来源冻结、12小算子同输入核对、point/RMS整层集成、全29阶段及CTA等待表完成。
- [x] CPU补齐短/中/长KV、12原语/两进程与完整层汇总；包括退化和波动。
- [ ] 下一步：先以相同前序访存/缓存条件定位Cross K/V，再查FFN执行资源差距；之后按适配tile和DAG就绪填空。
- minor-ops本轮已结束，GPU207.379/1800秒，M39累计1611.324/36000秒，旧截止保留。
- [x] M40最新授权：GPU1启动固定CURRENT18；原09:30截止保留，新窗口独立登记。
- [x] 已部署自动CURRENT完整审计→ON3000→部署CAL队列。
- [ ] 完成固定DEV90三臂入口/首两S审计核查，再接既定完整评估。

M39前轮累计1403.945秒；本轮新增207.379秒，累计1611.324/36000秒。M40累计5.7624/24小时。
BAAI仅~/cxz，AutoDL已归还。保护state/cycle/rules、baseline、原tile、LIBERO生产、四项false不变。

历史准备：[GEMM同前序条件诊断](../../experiments/single-layer-async-gemm-20261020/gemm-context-control-20261021/PLAN.md)，GPU≤600秒、CPU2，该准备时尚未恢复机器人；现已由新用户优先授权接续。

- [x] 同前序Cross KV/FFN两进程诊断完成（首失败保留），下一步设备函数/编译资源单因素。M39当前累计1649.673秒。
- [ ] 当前优先M40，CURRENT18正在GPU1运行；新截止与原24h余额见current18登记，四权限false不变。
