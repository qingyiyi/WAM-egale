---
kind: decision
name: 训练数据与接受标准
level: reference
scope: research
status: active
updated: '2026-10-18T07:48:45+08:00'
supersedes: []
sources:
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
- .pipeline/research/speculative-sampling/ACCEPTANCE-CRITERIA-CCFA-20261014.md
- .pipeline/research/speculative-sampling/TRACE-SUCCESS-CONTROL-RESULT-20261014.md
- .pipeline/research/speculative-sampling/OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md
- https://arxiv.org/html/2605.13778v1
- 投机采样/results/runtime/p0-target-trace-draft-train-v1-20261015-r1/PLAN.json
- 投机采样/results/runtime/p0-tf-effect-collect30-pack-cpu-v1-20261016-r1/MANIFEST.json
- 投机采样/results/runtime/p0-tf-effect-onpolicy-train-cpu1-v2-20261017-r1/PLAN.json
---

# 训练数据与接受标准

历史TRAIN194共179928 query，FIT174为159691行、CAL20为20237行，覆盖全部10个Long任务。FIT为各任务init0–17，除task0/1的init6/7/8缺6条，CAL为init18/19。不是只用task8或少数任务训练。target轨迹监督不等于当前TF访问分布；另14997混合行含3906 A_off与11091旧研究TF，不是新child on-policy。

TF/MLP各三seed，stage1 2000步(lr1e-4)，stage2加500步(lr3e-5)，batch512。两阶段选TF23/MLP22；CAL局部接受约95.0593%/95.4567%，不是任务成功率。真正on-policy聚合仍是待研究方案，未宣称已完成。

后续当前TF_ON3000另从CHILD2500固定500更新，batch512中target256加新FIT256，normalizer固定。新数据完整30局：全10任务×init30/31为FIT20，共11416行；init32为CAL10，共6805行。生成器是TF_CONT3000，不是当前TF_ON；只纳入root与实际reached parent，不把反事实全部节点称访问轨迹。不能把该元数据中的ON名字解释成当前模型已完成递归on-policy训练。

增加数据不直接保证成功率或trace长度。训练/CAL/DEV/TEST按完整episode隔离，评估样本回填训练后须披露暴露，不再称其独立TEST。本文只补记来源，不授权新增训练或替代旧预算/NO_GO。

当前K2/B2 bonus、prefix_then_cost、共享KV；接受分数 `abs(Δσ)*L2(vdraft-vtarget)`，112维，阈值.007839680526450805来自旧task8/B1的2400误差q90。它不是论文给出的机器人成功保证；packed/sequential BF16可有差异，τ0也非自动严格等价。

σ/timestep由scheduler关联；目前同时作为输入，不应假设独立。相同目标约束下greedy/DP等价不能作为新贡献。

接受标准依据、数据覆盖、闭环风险、真正新TF on-policy与必要条件输入需有限实验；不按已见失败task名单硬编码回退。

## PF周期的定义与证据边界

PF2是Spec、Spec、原Target三个policy chunk循环，PF3/4类推；bonus是chunk内额外去噪推进，二者不同。原Target chunk使用候选当时到达的真实观察，不回滚物理动作、不恢复原基线状态，也不等于清空历史KV。周期刷新不提供任务成功率或环境trace长度保证。

旧A_off小MLP在2026-10-14的task0/1×init13/14四个匹配DEV均成功；连续投机相对T最差变长20.44%，PF2为2.55%，但target调用相对连续投机增加18.40%。10/15另两个复用DEV pair，连续/PF2均2/2成功，PF2调用增加13.38%，未证普适质量优势；调用数不是墙钟加速。这些是旧模型/局部开发证据，不能借给当前TF64。

FLASH预印本v1表3确实测试周期full-path：无FB时PF2成功率80.6%、PF3为82.0%；带阶段fallback时PF2组合84.6%，不能说PF2始终最优。其π0 flash path复用旧视觉KV，full path另刷新该条件，与LingBot当前action去噪投机不同；不能直接继承其收益或当作已核实A会录用。

PF是独立算法配方维度，报告须明确开关/周期、固定checkpoint和接受器。启用后按新版本前瞻验证，保留PF-off原失败，不修改旧质量字段。本文仅为level: reference知识，不表示用户决定启用PF、不授权新GPU任务或周期搜索；当前启用状态以实际配置/作业为准。
