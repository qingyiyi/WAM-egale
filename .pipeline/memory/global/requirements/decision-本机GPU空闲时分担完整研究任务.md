---
kind: decision
name: 本机GPU空闲时分担完整研究任务
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-17T19:32:00+08:00'
supersedes:
  - '将本机减负理解为新增实验全部只能等待AutoDL的资源安排；本机空闲时用户明确允许分担'
sources:
  - '用户：目前本机gpu如果空闲的话也可以分一部分任务在本机先跑着'
---

# 本机空闲GPU可分担当前已批准的完整研究任务

用户明确允许在本机GPU空闲时分配当前研究的一部分任务，加快现有AutoDL工作。先核对GPU、CPU、显存与存储，只调度尚未开始的完整任务或预登记分片；不停止或重复其他机器已经执行的episode。新分配保存硬件、数据来源和原登记的起点、截止、尝试及空间上限，不重置旧作业预算。

相同初态的对照臂固定同一物理GPU，来源和硬件分层明示；不混本机SXM4与AutoDL PCIe探索计时。全臂继续完整512-token BF16优化reset，不改变模型结构、权重、阈值、训练配方、旧失败或四项false。资源竞争时只管理自己的任务，完整episode边界收尾并记录PARTIAL/暂停，不杀他人、不改系统；正式性能仍需独占和原质量准入。

本决定只细化[本机减负偏好](preference-本机压力时优先远端计算与容量告知.md)，本机繁忙时仍优先现有AutoDL；[暂不扩容](decision-本轮AutoDL不扩容并先清理废弃资源.md)不变。
