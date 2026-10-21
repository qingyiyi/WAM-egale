---
kind: decision
level: constraint
scope: wam-research-20261016/M40;仅PARENT采集place_object_basket/valid08
status: active
updated: '2026-10-21T08:42:05.494082+08:00'
source: 用户明确允许补跑此局、保留原PARTIAL；随后明确允许新的最多60分钟窗口，累计预算不增加
supersedes: 此一局的补跑待答复与新窗口待答复状态；不修改任何原作业截止或终态
---
# 允许补跑RoboTwin parent valid08，并新增六十分钟窗口

仅补跑原因重复terminal forward在0环境动作退出的place_object_basket/valid08，使用修复并通过r3八完整policy门的同一PARENT2000路径。原PARTIAL、原截止与全部消耗保留，不覆盖旧尝试，不改变固定FIT/CAL。

用户已明确批准独立新Attempt：从本次实际启动起最多60分钟，且最晚北京时间2026-10-21 10:00清理。包含加载、等待、完整episode、失败与清理，计入原24 GPU生命周期小时，累计预算不增加；不自动重复失败尝试。原per-episode1800秒、8000teacher和清理900秒继续适用。

此许可不扩大DEV/TEST，不修改四质量权限、阈值、权重或环境。固定18采集齐并审计通过后，已授权的后继训练仍须遵守各自原预算和夜间截止；本单局新窗口不自动延长其他作业。
