---
kind: feedback
name: Transformer子图设计应覆盖self与cross-attention
level: guideline
scope: cycle:wam-research-20261016;operator_design
status: active
updated: '2026-10-19'
supersedes:
  - M30前版将FFN-first建议当作整体Transformer推进路线的表述
sources:
  - 用户2026-10-19指出前版拆分过度照搬MegaMoE，要求根据Transformer的self与cross attention考虑tile子图
---

# 按Transformer真实结构设计

整体tile/persistent方案应依据当前Transformer的self-attention、cross-attention、FFN及归一化/残差依赖展开。FFN可以作为局部实验，不能代表完整Transformer的调度设计，也不应仅因MegaMoE参考以FFN为中心。

明确attention内部tile计算和完整attention子图的区别，说明QKV准备、在线softmax、输出投影与后继节点的就绪条件。MegaMoE只提供任务组织和常驻执行思想，不能以其模型分解替代WAM结构。

本条保存用户对整体设计范围的反馈；具体tile尺寸、实现顺序、warp角色和收益仍是待讨论/验证方案，不把用户问句记为已批准某个GPU实验。既有代码保护、数值门、资源预算不变。
