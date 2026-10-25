---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;本机A100测试
status: superseded
updated: '2026-10-25T13:47:40.038498+08:00'
sources:
  - 用户2026-10-25明确要求本机A100测试先暂停，等其通知再继续
supersedes:
  - 本机M42多初态评估的自动接续许可；不撤销远端A100和5090授权
---
# 本机A100测试暂停，等待用户明确恢复

暂停本机A100上的项目测试，清理本轮模型服务与模拟器，禁止守护或后续会话自动恢复。只有用户明确通知继续才可接续；原预算起点与15:45:32截止不自动延长。

远端两张A100的RoboTwin和5090算子实验不在本次暂停范围。保留所有完整与部分结果，用户中断单列，不计作自然算法失败。无需为暂停改写旧FAILED或科学权限。

执行证据：`.pipeline/experiments/suite-acceptance-calibration-20261025/multi-init-20261025/prep02/USER-PAUSE-RESULT.json`。

用户随后于2026-10-25T13:59:10.198632+08:00明确恢复本机工作。本次暂停历史保留，恢复见decision-本机A100按原截止恢复固定Goal清单.md。
