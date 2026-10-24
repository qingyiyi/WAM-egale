---
kind: decision
name: 仅补齐明确授权的基础设施失效任务
level: constraint
scope: cycle:wam-research-20261016;M16;M17
status: active
updated: '2026-10-18T01:25:24.688361+08:00'
supersedes:
  - '对以下明确列出的失效case不自动重试；用户已逐项授权新namespace补齐'
sources:
  - '用户：允许补齐8个迁移动作前失效Long case（三臂24局）和Object case0；全部原PARTIAL保留'
  - '用户：允许补齐case1，保留原PARTIAL（Object新入口首动作前顺序检查失效，0 teacher forward）'
---

# 只补齐已明确批准的失效case

Long只允许case_id 219、220、221、222、223、225、226、227，各T/C/P共24完整局；Object T_POOL允许case0和case1各1局。使用独立namespace，保留初态、评估seed、权重、threshold、完整horizon、reset优化和全部原数学/审计。原失败与新attempt身份分列，同一case/arm在成功率面板只占一个预登记位置，不作为新的独立初态。

所有失败、准备、等待、清理及新实际forward仍计原M16/M17预算与绝对截止，不重置账本。其他完整算法失败不得按此名单重跑；其他基础设施失效也不自动借用此次许可。其余从未开始任务继续运行。

用户随后允许清理程序/环境错误的无效产物，具体保存/删除规则见[仅清理失效产物](decision-清理基础设施错误产物但保留有效任务失败.md)。当前运行中引用的原件须先保留，不能删断审计依赖。
