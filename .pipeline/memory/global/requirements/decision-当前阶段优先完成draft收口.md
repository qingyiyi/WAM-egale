---
kind: decision
name: 当前阶段优先完成draft收口
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-16T19:18:00+08:00'
supersedes:
  - '当前Cycle中将新增系统实验与draft收口并列优先的安排；只替代资源优先级，不替代研究因果主线或已完成证据'
sources:
  - '2026-10-16用户：我想优先把draft model搞定'
---

# 当前阶段优先完成 draft 收口

本阶段优先确定 draft 的结构、训练方法及闭环可用性，再扩大新增 MegaTransformer 实现工作。保留 TF64 主候选和等容量 MLP 对照；偏好 Transformer 不替代质量门。

先诊断再训练的既有规则继续有效。与 draft 可用性直接相关的 target/verifier 数值排查属于本优先任务；暂缓新增纯性能原型和大融合，不修改原 tile。诊断结论不写成本决定中的已证实根因。

原 FAILED、retention、threshold、四项 false 权限及独立 TEST 隔离全部保留。此优先级决定不自动授权 120 局、架构搜索、新租卡或第二环境；实际作业仍须事前登记资源边界。
