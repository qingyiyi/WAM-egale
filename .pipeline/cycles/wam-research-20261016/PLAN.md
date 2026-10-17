---
kind: plan
cycle: wam-research-20261016
status: active
updated: 2026-10-16T22:23:00+08:00
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM 投机与多候选系统协同研究

> 10/17早当前实验已按用户要求暂停并清理GPU。DEV72/96、TF-on新增风险失败，见本轮RESULT；先做网页版文献/开源交接和GitHub发布，不依据历史计划继续启动。

## 目的与边界

用户已返回完整执行方案并授权研究继续。原夜间AutoDL09:00授权已经结束，晨间归还用户后不再SSH。当前50完整DEV已结束但TF retention HOLD，最新事实superseded_by [今天结果](../../research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)。主线及有限工作包统一见[执行索引](../../research/speculative-sampling/WAM-COORDINATION-20261016.md)，本Cycle仅保存恢复状态，不复制长计划。

## 完整计划

| ID | 阶段 | 期望结果 | 验证方式 |
|---|---|---|---|
| `M1` | 保留本机闭环终态 | 准确记录TF研究候选与质量未过 | 原fresh30、parent10、统计协议及终态报告 |
| `M2` | 既有AutoDL任务与接续 | 旧100归档，已授权新链按原预算运行并如实收尾 | 真实status、结果、清理和预算，不重复启动dispatcher |
| `M3` | 核对网页端总计划 | 将有依据的决策/任务/验收/日期纳入HW | 用户返回的计划与本地证据逐项核对 |
| `M4` | S01/S02 真实形状、强基线与机制 | 1/3/7状态、短中长真实输入与算子账本 | 实测与接口推导分开；无新TF正式性能 |
| `M5` | Q02 质量风险与数据隔离 | 五假设审计及有限反事实准备 | 缺原始字段明确未知；不解盲TEST |
| `M6` | S03 原型选择与收缩 | 有profile支持的一条主路线 | 相对已有最佳路径；逐级正确性门 |
| `M7` | 夜间归档与晨报 | 终态、资源清理、关键路径 | 原FAILED保留；不把子图结果写成机器人收益 |
| `M8/M9` | 晨间补训与80完整DEV | 已完成描述证据，保留原PARTIAL | 不替代严格证书 |
| `M10` | 同预算MLP及前瞻50完整DEV | 已完成，风险新增失败，TF未retained | 3184PT全raw独审、来源/预算/58pins清理真核 |
| `M11` | 前瞻风险日志诊断 | 1对完整、1个Target重复、1个TF资源中断；当前权重NO-GO | PARTIAL原失败保留；下一步同query/全KV数值复验 |
| `M12` | 同query教师路径定位 | 两真实记录chunk及时间编码反事实完成，11.72分钟内清理 | B2/B14差异从FP32时间投影出现，单独规范时间未恢复最终方向；不冒称整局根因 |
| `M13` | TF64唯一一次数据配方收口 | 按用户授权1→2→3：诊断、FIT20/CAL10、等预算训练、96完整DEV | [独立实验与运行接续](../../experiments/tf-effect-closure-20261016/RUNBOOK.md)；新质量结果未获，旧四权限false |

## 验证与未决

## 当前用户优先级（2026-10-16晚）

用户明确优先完成 draft 结构、训练方法与可用性收口，见[决定](../../memory/global/requirements/decision-当前阶段优先完成draft收口.md)。新增系统性能原型暂缓；直接影响 draft 质量的同 query 数值诊断优先，不据偏好放行当前权重。

- [x] 保存用户优先级，恢复现有证据及当前资源。
- [x] M12事前登记：同真实根query、两完整记录policy chunk、完整条件与30层有效KV；GPU墙钟20分钟/1GiB，旧预算关闭不重置。
- [x] M12执行与归因：两记录chunk、同query与时间编码反事实完成并清理，见[结果](../../experiments/draft-teacher-path-20261016/RESULT.md)。不是新episode、不触发训练。
- [ ] M13 TF效果收口：主结构固定TF64；先核当前TF递归日志与训练来源缺口，证据支持才一次新FIT/CAL及TF-cont/TF-on公平训练；MLP后置作baseline，见[下一步](../../experiments/draft-teacher-path-20261016/NEXT.md)。
- [x] M13第一步CPU覆盖审计：904树/2712真实query与历史同task同step来源核验完成，33.83秒、0GPU；[结果](../../experiments/tf-training-support-20261016/EXPERIMENT.md)。不能据简单尺度统计强行触发训练。
- [x] 按诊断证据触发唯一一次等预算训练：已完成一次新FIT20/CAL10完整采集、全量审计和固定三条件500更新；原CPU失败保留，GPU接口与DEV仍待实际验收。
- [ ] 用前瞻登记的完整任务确认候选；不修改旧 FAILED，不自动启动 120 或独立 TEST。

2026-10-16 21:18：用户授权完成1→2→3；M13新增匹配step误差/联合覆盖CPU诊断57.14秒完成，支持一次有限当前到达分布实验，不证明整局失败根因。执行清单及准入解释见[TF效果收口](../../experiments/tf-effect-closure-20261016/PLAN.md)。完整50chunk collector OFF/ON资格r2正在本机共享GPU运行；r1仅监督父PID比较错误、0GPU FAILED已保留。两Attempt共用原21:49:55清理硬帽；未采新30、未训练。

成功率工程门尚未通过，新TF正式性能未测；今天50完整但原风险条件失败，未来120真实门拒绝。结构接口暂冻不等于权重/配方accepted，四权限false、原FAILED和默认不变。夜间09:00/10:30条款仅为已关闭历史链，午间独立4h已收尾，不能重置续跑。后续作业先登记，候选/引擎冻结后才TEST40–49。中文稿由用户整理飞书。
