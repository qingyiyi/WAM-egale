---
kind: decision
name: 优先恢复RoboTwin并占用远端一张卡
level: constraint
scope: cycle:wam-research-20261016;M40
status: active
sources:
  - 用户2026-10-21询问两卡空闲并明确要求优先RoboTwin迁移占一张卡
supersedes:
  - 旧晨间窗口结束后尚未获得后继接续授权的当前等待状态
---
# 优先恢复既定迁移链

用户现已明确要求继续RoboTwin迁移，优先分配远端一张GPU。从未开始的CURRENT18接续已有CONT3000，完成监督审计后执行既定ON3000、接口校准和固定DEV，依赖不成立则停。GPU0可并行已授权算子工作，CPU核隔离。

本次为新的有界接续窗口，原24小时累计GPU预算不增加；保留每个旧deadline、PARTIAL和失败，不重写旧窗口。仍只覆盖既定六任务迁移，不启动全50/Hard、独立TEST或正式性能，四科学权限不变。实际新窗口、清理预留与任务预算写在新作业登记。

用户进一步明确RoboTwin继续后台运行。保留既定GPU1采集→审计→ON训练→接口CAL自动接续；算子在GPU0推进，不因算子讨论暂停机器人。依赖、原截止和累计预算仍约束后继，不将此解释为无限重启或扩大测试。
