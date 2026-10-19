---
kind: preference
name: MegaMoE作为tile调度思想参考
level: guideline
scope: cycle:wam-research-20261016;operator_tile_persistent_design
status: active
updated: '2026-10-19'
supersedes: []
sources:
  - 用户2026-10-19建议tile化和persistent warp参考MegaMoE
  - 用户2026-10-19明确只需完成MegaMoE思想，并要求综合HW优化设计
---

# MegaMoE参考范围

本轮重点借鉴MegaMoE的tile工作分解、常驻执行、warp角色与生产者消费者依赖调度思想，不要求复刻其官方实现。设计应结合WAM的dense Transformer、小action形状和共享/私有状态，不为模仿MoE而引入无关路由、通信或精度变化。

阅读固定版本官方原文/源码，区分实际机制与本项目建议；面向A100/SM80重新设计适配，不能默认新架构专有原语可用。具体任务图、tile形状、调度策略和性能承诺仍属待验证研究，不记录为用户已确认结论。

交付包含整体设计、依赖、最小可证伪原型和验收，不仅提供相关仓库链接。实验预算、质量门和源码保护不因参考偏好而扩大。
