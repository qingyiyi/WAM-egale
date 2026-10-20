---
kind: decision
name: 优先修复RoboTwin环境且保持系统驱动
level: constraint
scope: cycle:wam-research-20261016;robotwin_pilot
status: active
sources:
  - 用户2026-10-20要求先解决RoboTwin不能运行的问题，不随意改驱动，核查A100适用性及是否需要4090
supersedes:
  - decision-暂缓RoboTwin并优先收口算子结果.md中暂停环境排查的当前优先级
---
# 恢复环境排查，保持系统驱动

用户已明确恢复并优先要求解决RoboTwin启动故障。允许在已有资源和用户目录内检查、修复运行环境，核验A100与RTX显卡的适用性；系统驱动保持不动，不自动租用或购置新GPU。

此决定替代前条暂缓环境排查的安排，不追溯修改旧失败或原预算。先复现原始渲染/场景并验证有证据支持的兼容修复，不把替换显卡视为已批准采购，也不把仅环境通过当成模型成功率。

用户同时要求对照MegaMoE思想审查当前算子的完成项、缺口、慢因与下一步，和RoboTwin结果一起汇报；本次算子部分为源码与既有证据分析，不自动扩大算子GPU实验。
