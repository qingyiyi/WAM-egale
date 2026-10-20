---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T01:45:15.402937+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

沿既定用户授权连续执行，不新建人工审批阶段。夜间[安排](OVERNIGHT-PLAN-20261020.md)至09:30–10:00，旧作业截止/预算不重置。

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | 四套LIBERO证据收口 | 6000局原质量/长度/成本与审计缺口；不追加TEST |
| `M39` | 简易单层先对齐再调度 | 同输入逐阶段原版/独立/整层对照；9签名/全部有效KV/RNG有限逐位；解释仍未对齐原因 |
| `M40` | RoboTwin固定迁移链 | Target48→parent18→current18→固定三臂DEV90，分区/原失败/预算保留 |

## M39 当前执行顺序

- [x] 简易版与复杂版同输入比较，无稳定退化；复杂活动路径归档，后续仅维护optimized简易whole_layer。
- [x] pointwise/layout、Norm搬运、RMS双行、FFN1 M128、cross连续KV加载、B14 Q16使用M64 attention逐项有限核验。
- [x] 更新当前阶段矩阵；N256 FFN1精度通过但整层变慢，已撤回。grid_constant原已存在，取消冗余试验，0GPU。
- [ ] 依最新矩阵有限改进pointwise tile与剩余原语，完整层复核；无法对齐给可追溯理由。
- [ ] deferred：原语对齐完成后才考虑跨阶段ready；当前不启动调度，阶段grid.sync保留。

[当前阶段矩阵](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/stage_final/RESULT.md)。测量是uncached/eager单层，不是强cache/Graph、30层或机器人E2E。M39总10GPU生命周期小时、本夜3小时帽，失败/清理计入；阶段调用帽12000，最新前瞻block帽5600。

## M40 当前执行顺序

- [x] Target48完整，41成功7自然失败；监督FIT36/CAL12审计齐，PARENT2000和一次CAL完成。
- [x] 保留FSDP2类型失败、旧接口有限通过和parent首局0动作PARTIAL；加强门r3通过8完整policy核验。
- [ ] 原parent剩余17运行，原03:55绝对截止不改；缺place_object_basket/valid08补跑请求待答复。
- [ ] 固定parent FIT12/CAL6齐后才能接续原训练，不将11FIT改称完整12；不因结果替换样本。
- [ ] 原current18与DEV90尚未启动；沿原174局/24GPU生命周期小时/分区/失败规则。

baseline、LIBERO投机、原tile与保护HW不改，BAAI限~/cxz。四权限false。10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00证据包。[旧计划](history/alignment-m64-20261021/PLAN.md)保留。
