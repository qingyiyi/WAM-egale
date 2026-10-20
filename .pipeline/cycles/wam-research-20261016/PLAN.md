---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-20T21:59:48+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M39
    status: in_progress
  - id: M40
    status: in_progress
---
# 当前有界计划

## M17 — 四套LIBERO证据收口

6000局计算齐，旧科学审计未放行。保留曝光/首两S/失败/长尾，不追加未登记TEST或正式性能。

## M39 — 当前单层开销与下一优化

- [x] context128前轮有限门与普通完整层计时，B14三KV均小于同轮Torch两倍。
- [x] 当前实现24完整block/150原语有界profile，阶段与native核对，源码和资源终态审计。
- [x] 解释tile、warp、队列、同步及性能边界，给出依据当前数据的排序。
- [ ] 先完成同输入/同边界原版、自有阶段独立、自有阶段嵌入整层三列对齐，区分原语和整层执行成本。
- [ ] 再优化Norm/pointwise等主要差距；保留归约与舍入后逐项验证。
- [ ] FA2嵌入开销定位/专用化，再context K/V真实ready调度；不先大范围修改所有阶段。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)，[资源](../../experiments/single-layer-async-gemm-20261020/context128-profile/RESOURCE-AND-STOP.md)。本轮未改数学源码。原1497/1500、372/420与20:20:28截止保留；最新用户明确要求先分析，新profile登记独立24/150、900秒、22:11:12截止，已完成。后续GPU实现需独立有界登记，不把旧余额当无限授权。

## M40 — RoboTwin固定迁移链

- [x] 首六Target自然终态5成功1失败、114初态、108完整bank。
- [x] 首六4150监督query/原固定42清单CPU审计，独立wrapper与有界guard准备。
- [ ] GPU1自动完成剩余42条Target；现场10/20 21:59:00北京：新增5/42完整终态、5成功，当前`open_microwave-valid01`；原首六5成功1失败保留。
- [ ] 合并Target48的FIT36/CAL12，审计后接固定TF适配/采集/训练/DEV；当前guard不自动训练。

[登记与计划](../../experiments/robotwin-pilot-20261020/target-remaining42/PLAN.md)：6小时job生命周期，21:37:02至次日03:37:02，4GiB新增，逐局1800秒/8000teacher、开局预留整局和900秒清理；累计原24GPU小时帽。非infra重试、不截断episode、不补造旧收据。latest continue替代本轮只观察限制，旧截止/账本不改。

M31–M38已完成及旧M39计划见[历史](history/context128-profile-start-20261020/PLAN.md)。baseline/投机/原tile/保护HW不改，BAAI写入仅~/cxz。四质量权限false；10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包不变。

用户最新反馈落实为[逐阶段对齐优先](../../experiments/single-layer-async-gemm-20261020/context128-profile/STAGE-ALIGNMENT-PLAN.md)，当前仅计划，未追加GPU作业。
