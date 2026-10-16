---
kind: requirement
name: 实验与证据边界
level: constraint
scope: project
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
---

# 实验与证据边界

- 不运行研究 smoke、截短完整任务或独立大文件 SHA 作业；完整 episode 沿用登记的 horizon，预算不足报告 PARTIAL/FAILED 和未完成清单。
- 原 FAILED、负结果、原始协议及真实输入保留。后续统计分析与新协议应独立命名，不能追溯把原失败改 PASS，不事后挑 seed、阈值或成功子集。
- 连续动作投机是近似算法，不能宣称离散 token 投机的严格分布保持。局部接受率不等于任务成功率，成功局步数不等于所有任务质量。
- 同算法工程优化要核对输出、接受路径、scheduler/RNG 和全部有效 KV；独立算子的 allclose 不能自动替代这些要求。
- 正式计时须无 GPU 争用，注明 kernel/block/verifier/chunk/task/cold E2E 的边界。性能、质量及数据采集结果分别报告，不能把更快失败当收益。
- 修改 state/cycle/rules 保护源、原 tile、现有实验冻结源或服务/系统环境不属于此次 HW 维护。
- 仅清理有明确依赖与可恢复性结论的内容。失败目录不等于无用；凭据不进 HW。
