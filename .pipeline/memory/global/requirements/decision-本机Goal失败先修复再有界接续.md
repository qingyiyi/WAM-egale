---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;本机M42接续
status: active
updated: '2026-10-25T15:56:53.159109+08:00'
supersedes:
  - decision-本机A100按原截止恢复固定Goal清单.md 的当前执行窗口限制；原记录/账本保留
sources:
  - 用户明确“另外本机工作你需要继续完成，失败了分析原因，正确跑起来”
---
# 本机Goal失败先修复，再有界接续

用户要求分析本机失败并修正后实际接续。原准备失败、旧15:45:32窗口及所有结果保留，不把旧窗口改写为新授权。新作业先登记起点/截止与资源上限；沿用固定Goal多初态与阈值配方，质量/长度/接受深度为主，耗时附带。此次不因启动失败就放弃后台目标，也不能只报告过时RUNNING状态。

修复限定资产布局、运行包装与owned清理，不放宽严格初态/质量门，不自动重训或扩套件。当前有界登记落点`.pipeline/experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/PLAN.md`。具体6小时是本作业采用的上限，不提升为账号无限资源授权。
