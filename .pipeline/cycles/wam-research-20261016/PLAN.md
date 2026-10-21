---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T10:46:42.181704+08:00'
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
- [ ] 下一步：用当前表定位Cross K/V与FFN执行资源成本；每算子按适配tile/任务数，未来按DAG就绪填空，不凑108。
- minor-ops本轮已结束，GPU207.379/1800秒，M39累计1611.324/36000秒，旧截止保留。
- [ ] M40下一窗口：CONT3000 CURRENT18→审计→ON3000→固定DEV；09:30原窗口不自动延长。

M39前轮累计1403.945秒；本轮新增207.379秒，累计1611.324/36000秒。M40累计5.7624/24小时。
BAAI仅~/cxz，AutoDL已归还。保护state/cycle/rules、baseline、原tile、LIBERO生产、四项false不变。
