---
kind: knowledge
name: 投机去噪近邻与创新边界
level: reference
scope: project:WAM-egale;related-work-checked-20261019
status: active
updated: '2026-10-19'
sources:
  - .pipeline/experiments/speculative-denoising-novelty-20261019/RESULT.md
  - .pipeline/experiments/speculative-denoising-novelty-20261019/PRIMARY-SOURCES.json
---

ICML2025 Accelerated Diffusion Models via Speculative Sampling、ICLR2026 FlowCast、TS-DP v1、FLASH v1已覆盖连续投机/velocity门、单层Transformer蒸馏、机器人批量verify和周期PF等组成。TS-DP/FLASH正式venue本轮UNKNOWN。TS-DP的NFE比例不能称任务E2E；FLASH直接draft动作chunk/复用旧VLM上下文，不等同本项目chunk内去噪树和action-only SST。

Sequoia/Cascade已有硬件成本树优化及shared-prefix执行。新近Speculative Forcing用于Lingbot-World2.0视频，WAMJET已对LingBot-VA做常规缓存/Graph/融合优化；不称这些方向未被研究。RIFT/1Step-WAM为不同WAM加速近邻，未在本机复现。

潜在新增机制仍是HYPOTHESIS：在最强B14上利用合法候选共享/私有状态的Transformer执行组织，证明额外收益和闭环有效提交成本。普通batching、copy和FFN融合单独不足。必须做形状×引擎/缓存因素消融、sum(T_round)/sum(g)和质量；检索不构成穷尽新颖性保证，四false不变。
