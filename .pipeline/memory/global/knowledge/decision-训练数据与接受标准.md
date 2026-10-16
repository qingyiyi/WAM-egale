---
kind: decision
name: 训练数据与接受标准
level: reference
scope: research
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
- .pipeline/research/speculative-sampling/ACCEPTANCE-CRITERIA-CCFA-20261014.md
---

# 训练数据与接受标准

历史TRAIN194共179928 query，FIT174为159691行、CAL20为20237行。target轨迹监督不等于当前TF访问分布；另14997混合行含3906 A_off与11091旧研究TF，不是新child on-policy。

TF/MLP各三seed，stage1 2000步(lr1e-4)，stage2加500步(lr3e-5)，batch512。两阶段选TF23/MLP22；CAL局部接受约95.0593%/95.4567%，不是任务成功率。真正on-policy聚合仍是待研究方案，未宣称已完成。

当前K2/B2 bonus、prefix_then_cost、共享KV；接受分数 `abs(Δσ)*L2(vdraft-vtarget)`，112维，阈值.007839680526450805来自旧task8/B1的2400误差q90。它不是论文给出的机器人成功保证；packed/sequential BF16可有差异，τ0也非自动严格等价。

σ/timestep由scheduler关联；目前同时作为输入，不应假设独立。bonus与PF不同：PF2是Spec、Spec、原Target三个policy chunk周期；PF3/4类似。历史小样本不证明哪种周期最好，默认未启用PF。相同目标约束下greedy/DP等价不能作为新贡献。

接受标准依据、数据覆盖、闭环风险、真正新TF on-policy与必要条件输入需有限实验；不按已见失败task名单硬编码回退。
