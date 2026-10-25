---
kind: decision
level: guideline
scope: cycle:wam-research-20261016;M39-FFN1
status: active
sources:
  - 用户同意先对齐S的FFN1 GEMM，要求分析MegaMoE为何有效及当前缺口
supersedes:
  - A2负结果后继续扩大DSM交接块的当前优先级，不改历史授权与结果
---
# 先对齐FFN1基础计算再推进DSM调度

保留当前S，先按真实源码与分项证据核对FFN1加载/MMA、后处理和阶段等待，再决定一次有界优化。暂停凭warp角色或更大DSM交接块展开搜索。借鉴MegaMoE需区分负载、硬件、流水和调度条件，不以其结果保证本项目收益。各job独立登记并先正确性后性能，原负结果与四项科学权限保留。
