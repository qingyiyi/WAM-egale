---
kind: decision
name: 先核Robo分叉和成功长尾并补充chunk吞吐
level: constraint
scope: cycle:wam-research-20261016;M41-follow-up
status: active
updated: '2026-10-25T09:41:45+08:00'
sources:
  - 用户在成功配对、成本拆解及chunks每秒讨论后明确回复“可以的我同意”
supersedes:
  - preference-先核实整体加速并暂缓长尾深入诊断.md中对本轮Robo成功长尾继续暂缓的安排；只替代该有限范围
---
# 先核Robo分叉与成功长尾，补充完整policy吞吐

用户同意在当前工程测速结果基础上，优先核对RoboTwin首次PF2刷新前两臂已分歧的问题，再定位订书机等慢成功轨迹的额外动作；先区分实施或数值路径问题与算法误差，不直接归因为训练不足。

性能汇报补充完整policy的chunks/s及ms/chunk，明确分母是否包含独立cache/reset RPC、预处理和环境执行。PF2统计全部S/S/T，不只计投机chunk；同时保留有效更新成本、任务E2E、成功率、失败耗时和成功配对样本数。自然轨迹均值与固定输入独占复测分开。

原双A100固定600组三臂队列继续，原160 GPU小时及截止不重置；微波炉官方复现核验仍遵守其完整100组后的既定顺序。本决定不新增样本、训练、旧失败重试、独立TEST或正式性能权限。

用户关于逐任务阈值的本轮提问仍是讨论，尚不构成修改tau、接受分数、normalizer或生产候选的决定；相关只读核验见M41的ACCEPTANCE-CALIBRATION-AUDIT-20261025.md。旧质量门、失败、完整成本与原始记录保留。
