---
kind: decision
name: 双路线与算子保护
level: constraint
scope: project
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/tile-sdpa/PYTORCH29_FINAL_20261012.md
- .pipeline/research/speculative-sampling/MEGATRANSFORMER-TILE-INTEGRATION-20261014.md
---

# 双路线与算子保护

共同目标是有质量约束的 WAM 投机推理加速及 verifier 系统优化；投机与原 tile SDPA 的代码、证据和完成状态分别保留。

此次维护不写原 tile 代码或 `.pipeline/research/tile-sdpa/`，不恢复用户要求撤回的逐点融合。MegaTransformer 是用户受 MegaMoE 启发的自研方向，目前没有已完成的新融合原型或任务收益。

可规划真实 verifier profile 和接口设计；新增算子实现仍需落入用户确认的后续范围。新版 HW 安装不等于授权开始新的 GPU 研究。
