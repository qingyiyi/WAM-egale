---
kind: decision
name: 算子远端实现验证与本机最终E2E
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
updated: 2026-10-19T19:52:29.667745+08:00
supersedes:
  - decision-暂不迁移开发工作且本机模拟器由用户终止.md中M23暂不进一步迁移开发的阶段安排；用户自行处理旧模拟器的历史不改
sources:
  - 用户明确：先放到远端去实现，最后E2E性能在本机测
  - 用户随后明确：先实现tile化和persistent kernel，实现完汇报
---

# 远端实现算子，本机完成最终E2E性能

本轮将已授权算子开发、构建与正确性验证放到BAAI远端隔离目录，所有写入仍只在`~/cxz`。本机保留源码/HW/GitHub同步副本，最终E2E性能在本机获准的无争用窗口测量。当前优先实现tile与persistent kernel，再交付源码解释和实际验证结果；不以该优先级省略必要数值、状态与资源门。

远端既定Object评估继续，不修改投机代码或服务。算子baseline及实际依赖冻结，仅optimized变化；两机结果分别标注环境，不能拼接绝对时延冒充同硬件实验。局部算子计时不等于最终E2E，也不解除原四项false或自动启动受阻的闭环正式性能。

此决定替代此前本机拥塞排查时的暂不迁移开发安排；不是迁移整个用户工作区、购买资源、扩训练/TEST或改变模型数学的授权。具体卡占用、PID与ETA记实验，不作永久规则。
