---
kind: requirement
name: 资源与EGL并行规则
level: constraint
scope: project
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/rules/structured/project/data-collection-egl-parallel.yaml
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
---

# 资源授权与 EGL 并行规则

正式数据采集默认 EGL 和并行 eval；启动前验证 EGL、GPU、worker、服务/master端口、trace 与 provenance。原结构化规则不变：`.pipeline/rules/structured/project/data-collection-egl-parallel.yaml`。续跑复用已完成结果，按唯一键合并，不覆盖。

单 worker 必须记录实际资源约束理由。当前 AutoDL research 链固定单 GPU worker、约10核CPU配额，属于已记录的具体作业安排，不能推广成全项目取消并行规则。

本机正式性能用独占窗口；质量/算法实验允许按会话授权共享。用户10/15晚已恢复 GPU 与多核CPU，但不得据此忽略当前占用、累计预算、作业截止或启动新实验的范围。

不杀其他用户进程，不改系统服务/时钟。每项作业有墙钟预算、失败终态及自有进程清理。AutoDL租期和新链4小时/10月16日10:30限制属于该 Cycle，禁止自动续租或重置预算。

启动前先读动态 status；当前 HW 维护不启动 GPU、不重启或修改已有 dispatcher。
