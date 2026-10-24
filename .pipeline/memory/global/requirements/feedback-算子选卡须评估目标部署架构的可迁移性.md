---
kind: feedback
level: guideline
scope: cycle:wam-research-20261016;operator-hardware-planning
status: active
updated: '2026-10-23'
sources:
  - 用户指出H系列上研究的tile和warp分工不能直接用于4090或5090D，并质疑shared memory差异
  - 用户随后询问4090与5090D如何选择
supersedes: []
---
# 算子选卡须评估目标部署架构的可迁移性

用户反馈：研究硬件与计划部署硬件的资源和指令差异必须在设计阶段考虑，不能把Hopper专用实现移往RTX理解成最后补一次性能测试。

后续规划应区分可复用的计算依赖、候选分组与调度思路，以及需要按架构适配的tile尺寸、shared布局、流水级数、warp职责和MMA/搬运指令。跨卡速度和正确性必须在对应实现上核验，不把Hopper上的收益直接外推到RTX。

此反馈不等于禁止Hopper研究，也不把“所有tile/warp思路都无法复用”写成事实。用户尚未选定4090、5090D、H20或H800；助手选卡建议保持建议身份，不构成租卡、迁移、新GPU实验或扩充预算授权。原无H100资源事实继续有效。
