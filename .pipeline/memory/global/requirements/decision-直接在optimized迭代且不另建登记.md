---
kind: decision
name: 直接在optimized迭代且不另建登记
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
sources:
  - 用户2026-10-20明确要求不考虑另一agent带读版本，直接改opt代码，别另外登记
supersedes:
  - decision-依据profile继续单层tile流水优化.md中为带读保留独立源码版本及每次另行登记的要求
---
# 直接在optimized推进当前单层算子

本轮算子迭代直接修改现有optimized目录，不为另一agent的源码带读冻结优化代码，也不因一次优化再创建新的实验登记、Cycle或批准流程。已有本轮M39记录继续收口；必要的输入、速度、精度、资源与失败证据写入现有记录。

baseline及其实际依赖、投机代码与全局配置仍保护；单层persistent目标、旧FAILED、数值门与四项质量权限不变。不因减少管理记录而隐瞒结果、抹除历史或者扩大GPU/机器人任务范围。另一agent可按当前源码继续带读，代码教学不构成本agent优化阻塞。

来源是用户直接明确指令；替代此前M39为带读维持独立入口的安排。历史源码通过既有Git提交和实验源manifest追溯，无需在活动代码目录中保留教学副本。
