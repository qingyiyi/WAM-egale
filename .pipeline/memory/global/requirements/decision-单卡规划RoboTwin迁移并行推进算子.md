---
kind: decision
name: 单卡规划RoboTwin迁移并行推进算子
level: constraint
scope: cycle:wam-research-20261016;resource_planning
status: active
sources:
  - 用户2026-10-20要求先规划远端其中一张GPU迁移投机侧到RoboTwin测试，并同时规划算子侧下一步
supersedes:
  - RoboTwin仅保留一般调研、不制定具体资源与迁移执行方案的旧建议
---
# 单卡规划RoboTwin，另一条线继续规划算子

本轮需交付具体的RoboTwin投机迁移方案：使用远端其中一张GPU，说明版本、接口适配、数据与训练、完整任务验证、时间及资源预算。同时给出算子侧接下来的顺序、验收与止损。

这是用户明确的规划范围，不是新增第二环境安装、下载权重、采集、训练或评测的执行授权。GPU编号、任务数、训练轮数及预算属于待执行方案，不能记作用户已批准或作业已启动。

原约束继续：BAAI写入仅~/cxz；算子只改optimized、baseline冻结；现有LIBERO投机源码与旧证据不改。四项质量权限不因规划变化。原RoboTwin只读调研结果保留，旧“未安装/未执行”事实不被本记录改写。
