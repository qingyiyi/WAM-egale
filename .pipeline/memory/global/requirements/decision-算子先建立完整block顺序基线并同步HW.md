---
kind: decision
name: 算子先建立完整block顺序基线并同步HW
level: constraint
scope: cycle:wam-research-20261016;experiment:wam-block-baseline-20261019
status: active
updated: '2026-10-19'
supersedes:
  - 在完整block顺序基线尚未收口前直接开始persistent tile scheduler的安排
sources:
  - 用户提供WAM_BLOCK_BASELINE_FIRST_CODEX_PROMPT_ca5f37e_20261019.md并要求先推进、早上9点汇报
  - 用户随后明确：记得同步用HW更新算子侧部分的进展
---

# 完整 block 基线优先，HW同步更新

按固定科研快照ca5f37e审查真实替换/验证范围，优先在独立namespace建立R0原block、R1 attention替换混合block、B0显式分阶段混合顺序block。基线/设备原语与后续scheduler分开，不把attention替换或Python封装称为完整自有CUDA block。

附件默认CPU审查、准备和受限编译；未新增GPU、模拟器、训练、episode、迁移或生产修改授权。新GPU阶段须有明确有效授权，不借旧预算。保留原tile/保护源/旧失败与四项false。可继承旧有效证据，不盲目重跑或为PASS改标准。

本轮按北京时间10月19日09:00准备晨报。同步更新HW算子namespace，明确已观察事实、新实现、实际运行检查、PENDING/UNKNOWN与下一步；不能仅在聊天里声称进展。GitHub仍仅main作为公开镜像，不上传大tensor、权重或凭据。临时负载/进度和实际结果写实验，不写长期事实。
