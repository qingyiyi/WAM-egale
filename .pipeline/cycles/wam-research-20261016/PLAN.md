---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T02:34:36.439687+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 6000局原质量/长度/成本与科学缺口，不追加TEST |
| `M39` | 简易单层先对齐 | 原生/自有独立/嵌入整层对照，有限逐位及剩余差距理由 |
| `M40` | RoboTwin固定迁移链 | Target48→parent18→current18→固定DEV90，分区/失败/预算保留 |

## M39

- [x] 简易版与旧复杂版有限核验，后续仅维护简易入口。
- [x] 逐因素改进搬运、RMS、GEMM tile和FA2；精度与两进程完整层均核验。
- [x] 最新三种KV逐阶段矩阵与同SM时钟辅助观测完成；负结果撤回。
- [x] [阶段报告](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md)：有限精度通过，速度仍未对齐，已明确观测与UNKNOWN。
- [ ] 下一轮仍优先GELU/table向量化与FFN2原生执行组织；先阶段、再整层。
- [ ] deferred：达到对齐结论前不新增跨阶段ready调度。

当前不运行新GPU作业。本夜5970/6100 block、13572/14000 stage；
GPU生命周期619.984秒/3小时，M39累计约1285.849秒/10小时。
算子结论限uncached/eager单层，非强cached/Graph或E2E。

## M40

- [x] Target48、PARENT2000、一次CAL、加强完整policy接口门r3完成。
- [x] parent其余17已全部自然终态，原缺失局保留PARTIAL。
- [x] CPU审计固定18：FIT11/12、CAL6/6，未形成完整训练pack。
- [ ] place_object_basket/valid08补跑待此前明确答复，不自动重试。
- [ ] 完整固定FIT齐后才接续训练/current18/DEV90；当前均未启动。

原24小时/174局/1393500teacher、旧deadline保留；[报告](../../experiments/robotwin-pilot-20261020/OVERNIGHT-RESULT-20261021.md)。
baseline、原tile、LIBERO投机与保护HW不动；BAAI仅~/cxz，AutoDL归还。
四权限false；10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00证据包。
