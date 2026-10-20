---
kind: decision
name: 修复后接续RoboTwin有限迁移
level: constraint
scope: cycle:wam-research-20261016;robotwin_pilot
status: active
sources:
  - 用户2026-10-20要求robotwin先按照我们的方法迁移过去跑着
supersedes:
  - decision-暂缓RoboTwin并优先收口算子结果.md中的暂停迁移安排
  - decision-优先修复RoboTwin环境且保持系统驱动.md中的仅环境排查优先级
---
# 修复后接续原有界迁移链

用户明确要求将既定投机方法迁移到RoboTwin并运行。接续已批准的六任务计划：固定专家可解初态与FIT/CAL/DEV隔离，先原Target首六条完整任务，再按实际前置验收推进采集、固定训练链、部署校准及三臂DEV。

不重置原GPU生命周期、准备截止、存储和调用预算；不修改系统驱动、不启动额外环境或架构搜索、不改LIBERO源配置及旧科学结论。只有实际启动的阶段可写运行中，未实现的后继不能声称已自动排队。

本次提出的跨阶段就绪tile填尾波方案属于待分析设计，不将“能加速”或具体warp配比写成已确认事实或新的算子实验授权。
