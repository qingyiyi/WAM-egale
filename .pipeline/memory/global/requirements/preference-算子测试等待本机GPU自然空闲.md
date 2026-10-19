---
kind: preference
name: 算子测试等待本机GPU自然空闲
level: constraint
scope: cycle:wam-research-20261016;operator_GPU_validation
status: active
updated: '2026-10-19'
supersedes: []
sources:
  - 用户在整模型计时遭遇新GPU任务后明确选择“继续等待自然空闲”
---

# 算子测试等待本机GPU自然空闲

本轮算子整模型验证和性能计时继续等待本机GPU自然空闲，由已登记作业接续；不反复要求用户协调独占窗口，不终止或暂停其他用户进程。

等待计入本轮原墙钟截止，不重置预算。空闲后仍核对资源与输入；资源再被占用时保留实际部分结果，不将被打断的计时包装成正式结果。投机侧源码和正在运行的评估不受此安排影响。
