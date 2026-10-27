---
kind: decision
name: 研究范围扩展至flow-based机器人动作模型并候补StarVLA验证
level: guideline
scope: cycle:wam-research-20261016;同一篇论文的研究定位与后续跨模型实验
status: active
source: 用户2026-10-27明确要求记录到HW
supersedes: []
---

# 研究范围扩展至 flow-based 机器人动作模型

用户决定：论文的方法研究不局限于 WAM，可以面向更广的 **flow-based robotic action models**。FastWAM 仍是当前优先完成证据的主验证平台；若后续时间和资源允许，补充 **StarVLA-PI** 和 **StarVLA-GR00T** 上的方法效果，作为跨模型验证。

两项 StarVLA 实验列为当前工作的后续项，尚未启动，也没有结果。保留用户给出的模型名称；执行前再核对具体代码、权重、模型版本和动作采样接口，不预设它们与 FastWAM 的架构或 scheduler 相同。

跨模型验证沿用当前证据要求：先核对并建立各自 canonical，再适配教师方向、轻量预测器、接受阈值与合法 scheduler 提交；比较成功率、任务 trace、接受行为与完整成本。不能直接复用 FastWAM 的阈值或把单平台结果写成已证明的跨模型泛化。

本决定补充同一篇论文的后续范围，不替代当前 FastWAM 优先级，不中断已登记作业，也不扩大当前资源预算或自动启动新模型实验。只更新证据与受影响段落清单，不自动重写论文正文。
