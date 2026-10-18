---
kind: decision
name: 当前优先整模型算子Torch对齐并同步GitHub
level: constraint
scope: cycle:wam-research-20261016;operator-alignment
status: active
updated: '2026-10-18T21:00:51+08:00'
supersedes:
  - 本轮仅审查算子负结果、不实现独立修正版本的范围
sources:
  - 用户2026-10-18最新：主要负责对齐Torch算子，替换进模型后性能相近、精度一样；必要时复用Torch源码；同步HW和GitHub并提供网页审查prompt
---

# 当前优先完成整模型Torch算子对齐

当前主任务是实际定位和修复自有算子相对固定Torch2.9实现的差异，验证替换到真实模型后的输出、有效KV、RNG和性能。允许在独立新版本实现和直接复用固定上游源码，保留许可证及来源；旧tile基线和旧失败保留，不恢复被撤回的大融合。单attention容差通过不能替代完整模型一致性。

用户已授权将相关源码、研究进展和HW派生证据同步至qingyiyi/WAM-egale，并提供中文网页ChatGPT审查prompt。发布前排除凭据、大权重/张量与无关文件，固定新提交供审查，不覆盖旧科学快照身份。

此决定优先于旧只读算子范围；不改变旧FAILED、数值阈值、质量四false或全局默认。共享GPU只做精度/状态验证；正式计时需无争用。每个新作业登记独立有界预算，不重置M21或远端M16/M17原账本。
