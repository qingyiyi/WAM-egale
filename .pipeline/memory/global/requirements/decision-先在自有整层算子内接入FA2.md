---
kind: decision
name: 先在自有整层算子内接入FA2
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
sources:
  - 用户2026-10-20同意先在自己的算子内部实现FA2，并要求量化精度差异及调查RoboTwin迁移
supersedes:
  - 当前算子优先盲目放大GEMM tile或先移除全阶段同步的建议次序
---
# 先完成内部FA2，再研究跨阶段调度

先在optimized自有完整单层persistent算子内接入固定Torch版本的FA2设备计算，覆盖self-attention与cross-attention；不是在Python中调用外部SDPA替代整层实现。先保留必要阶段同步，用真实输入核对attention本身及整层数值、有效KV和性能，再讨论细粒度就绪队列。

用户要求给出现有精度误差的数量级及有证据支持的原因，区分已定位边界、数值归约假设和未解决根因。不能以FA2接入推定所有GEMM误差已解决。只改optimized，baseline/投机源与旧门保护，沿已有M39记录更新，不另建研究登记。

RoboTwin/LIBERO差异、任务数量与迁移成本为只读调研范围，不是新增环境安装、权重下载、训练或机器人测试授权。
