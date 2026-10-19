---
kind: decision
name: 算子tile化与persistent-warp调度为主线
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
updated: '2026-10-19'
supersedes:
  - decision-投机重组与多候选协同调度.md中10/16阶段仅主攻路线B、暂不展开路线A的安排
sources:
  - 用户2026-10-19明确同意继续，并要求tile化算子及persistent kernel基础上的tile调度
  - 用户2026-10-19要求综合HW研究当前算子优化设计
---

# 当前算子研究目标

范围修正（2026-10-19用户后续澄清）：本轮只做单层，整层统一persistent kernel内部调度全部tile，见[最新约束](requirement-单层Transformer统一persistent内核调度全部tile.md)。下文原30层推进文字仅保留历史，不再是本轮执行或完成门。

用户随后明确授权先实现tile化与persistent kernel，并要求完成后汇报；新M32工作先按真实primitive/完整block/30层门逐级实现与验证。

当前算子方向要研究可设备执行的tile原语，以及persistent kernel内的常驻worker/warp分工与依赖调度。普通host缓存、CUDA Graph、现成batching或只减少几个Python调用不能单独替代这一目标。

候选感知调度须说明相对现有最佳B14路径新增了什么；共享权重/历史与私有状态的边界保持，实际效益仍需验证。不能把用户指定方向记成已获收益或论文结论。

只保留baseline/optimized两份活跃算子实现；投机源码、原tile、旧FAILED和保护文件边界继续有效。先子图、完整block与严格状态门，再考虑30层与闭环；不因目标更新自动启动大型融合、新训练、TEST或正式性能。

此决定替代10/16仅主攻共享history attention的阶段优先级，不改写其旧结果、质量false、资源账本或历史截止。具体子图、tile尺寸、warp数量与预算由实际证据和有界作业登记决定，不是本条长期约束指定。
