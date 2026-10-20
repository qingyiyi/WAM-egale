---
kind: decision
name: 执行RoboTwin迁移与算子有限收口
level: constraint
scope: cycle:wam-research-20261016;robotwin_pilot;operator_research
status: active
sources:
  - 用户2026-10-20在两卡并行计划交付后明确同意按该顺序推进、做完汇报
supersedes:
  - decision-单卡规划RoboTwin迁移并行推进算子.md中的本轮仅规划、尚无执行授权条款
---
# 按两卡计划执行并汇报

用户已批准执行ROBOTWIN-AND-OPERATOR-PLAN-20261020.md所述有限方案：远端GPU1先复现RoboTwin原Target首6条完整任务，后接接口适配、84局分阶段采集/固定训练链及90局T/C/P配对DEV，GPU生命周期总上限24小时；GPU0依次做FFN2/Q/Norm2数值、原语效率和单层tile就绪流水，初始GPU生命周期8小时。按阶段实际结果收缩，不能预设通过。

授权包括该方案必要的~/cxz内独立环境、依赖、官方固定源码/权重/资产下载及适配代码；不包括系统安装/服务或驱动变更。CPU/存储/call/单局及清理帽见计划和执行账本，失败计预算，原账本不重置。

原Target复现不成立时先修复实施阻塞；算子数值未通过不接机器人。旧FAILED与四项质量权限保持，不能自动扩RoboTwin全50任务/Hard、无界训练、独立TEST或正式机器人性能。原LIBERO投机源码、baseline、原tile及保护HW不改；算子直接改optimized、沿M39保存证据。

本条只将本轮规划提升为执行授权，旧未执行事实及历史权限保留。完成后汇总真实进度、结果、失败、资源与下一步。
