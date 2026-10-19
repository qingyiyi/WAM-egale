---
kind: preference
name: 远端并行评估自动接续并聚焦本机算子
level: guideline
scope: cycle:wam-research-20261016;remaining_remote_LIBERO_evaluations
status: active
updated: '2026-10-19'
supersedes: []
sources:
  - 用户2026-10-19明确提出远端两卡各并行eval，并要求确认自动提交后主要考虑本机算子下一步
---

# 远端提高评估并行度，确认自动接续后聚焦本机算子

在现有两卡已授权完整评估中，优先核实每卡多个独立eval的可行性；显存峰值、CPU配额、上下文归属、端口、KV/RNG与输出隔离成立后，使用每卡最多两个独立worker提高吞吐。不能为并行而复用可变model服务或跳过资源准入。

任务依赖满足后自动接续尚未开始的固定分片，并核实真实日志和终态推进。确认队列可自行推进后，交互研究重点回到本机算子；远端保留低开销巡检。该偏好不新增科学臂、完整任务重复、租卡、原预算/截止重置或正式计时权限，也不授权修改投机算法源码。
