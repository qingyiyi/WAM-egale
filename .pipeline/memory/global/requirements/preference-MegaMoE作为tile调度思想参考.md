---
kind: preference
name: MegaMoE作为tile调度思想参考
level: guideline
scope: cycle:wam-research-20261016;operator_tile_persistent_design
status: active
updated: '2026-10-24T11:38:42.261583+08:00'
supersedes:
  - 本条原A100/SM80平台限定仅保留为2026-10-19历史阶段；当前算子平台按最新5090决定
sources:
  - 用户2026-10-24再次要求参考MegaMoE讨论TMA生产者warp专门化与mbarrier交接
  - 用户2026-10-23要求讨论FFN1/FFN2的TMA流水如何借鉴MegaMoE，明确先讨论
  - 用户2026-10-19建议tile化和persistent warp参考MegaMoE
  - 用户2026-10-19明确只需完成MegaMoE思想，并要求综合HW优化设计
---

# MegaMoE参考范围

本轮重点借鉴MegaMoE的tile工作分解、常驻执行、warp角色与生产者消费者依赖调度思想，不要求复刻其官方实现。设计应结合WAM的dense Transformer、小action形状和共享/私有状态，不为模仿MoE而引入无关路由、通信或精度变化。

阅读固定版本官方原文/源码，区分实际机制与本项目建议；原2026-10-19阶段面向A100/SM80；当前算子平台已由用户切换至RTX5090/SM120，须据该架构重新设计，不能照搬SM100的UMMA/TMEM或多卡通信实现。具体任务图、tile形状、调度策略和性能承诺仍属待验证研究，不记录为用户已确认结论。

交付包含整体设计、依赖、最小可证伪原型和验收，不仅提供相关仓库链接。实验预算、质量门和源码保护不因参考偏好而扩大。

用户在TMA输入交接讨论中再次强调生产者/消费者warp分工的参考偏好。应解释整CTA同步与按缓冲交接的区别，并基于当前SM120资源判断实现方式；具体“一warp提交TMA”的配置仍是讨论候选，不能写成已确认布局、已启动实现或已获得收益。

## FFN流水范围的后续授权

用户明确希望结合MegaMoE讨论FFN1/GELU/FFN2之间及CTA内部的TMA流水如何设计。应结合实际数据依赖、缓冲生命周期、warp职责和数值顺序解释可借鉴部分。此前“先讨论”阶段已结束；用户随后明确同意设计并允许容量压力下缩小tile，已在 decision-分步实现FFN的TMA流水并按共享内存约束调整tile.md 记录新的有界执行范围。用户同意设计不等于采用或性能成立；实际实现、数值和采用决定以当前Cycle实验产物为准。
