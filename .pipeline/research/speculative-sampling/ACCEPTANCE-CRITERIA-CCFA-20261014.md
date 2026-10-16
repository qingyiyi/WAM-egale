# 投机接受标准：CCF A 文献、当前阈值来源与论文表述

日期：2026-10-14，Asia/Shanghai。文献与来源复核；本轮未启动 GPU、训练或 LIBERO。现有 PF2 与四臂质量实验的结果仍以原报告为准。

**当前逐边公式有明确的一步更新偏差解释，也有正式会议中的相近接受规则；0.007839680526450805 是本地历史误差的经验分位数，不能写成文献给出的保质阈值。新找到的最直接参考是 FlowCast（ICLR 2026），其次是 ParaTAA 的原递推残差判据。**

会议级别按 CCF 2026 年第七版核查。CCF 官方有[正式发布公告](https://www.ccf.org.cn/Academic_Evaluation/By_category/)；本轮实际检查的是[中国科学院大学托管的 CCF 原目录](https://ai.ucas.edu.cn/index.php/zh-cn/jxjy/fzpy1/7675-ccf-2026)，72页PDF第57页将 ICLR、ICML、NeurIPS 列为人工智能 A 类。原件 SHA256 为 `271b630b576bf8a4f802e767f5694caded93680e22b3a19bef7902591c45c1d3`，页图与来源保存在下方证据目录。旧 CCF 分类网页与新目录可能未同步。

| 正式论文 | 实际判据 | 与本项目的关系 |
|---|---|---|
| [FlowCast，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)，§4、Eq.5、Algorithm1 | 沿用 anchor 速度外推未来状态；在 draft 状态重新查询 target 速度；速度 MSE 不超过阈值才接受连续前缀，拒绝处校正 | 最直接的确定性 flow 投机先例。本文用恒速外推，我们用训练的双头方向 MLP；本项目又乘实际 scheduler 步长，不能写成原算法复现 |
| [ParaDiGMS，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)，§3.1、Algorithm1 | 相邻 Picard 迭代的状态差满足 `||x_new−x_old||²/(D σ²)≤τ²`，才向前滑动窗口 | 支持按状态误差和前缀推进；它做迭代校正，我们当前只做逐边验证。ODE 实验仍借对应 DDPM 方差定标，不能把 LingBot 的 flow 时间 σ 直接当该方差 |
| [ParaTAA，ICML 2024](https://proceedings.mlr.press/v235/tang24f.html)，Eq.11 | 检查候选状态对原一步递推的残差，结合三角 Anderson 迭代 | 对“为什么检查原 scheduler 更新偏差”最有帮助；safeguarded 版本的有限步性质不能赋给当前 MLP 树 |
| [DPM-Solver，NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html)，附录D、Algorithm6–7 | 低阶/高阶更新差，经绝对与相对容差归一化，RMS≤1时接受，否则缩步重算 | 支持无量纲误差控制。将其归一化用于 draft/target 差属于适配；不是采用了原高阶求解器，也不继承其阶数保证 |
| [Accelerated Diffusion Models via Speculative Sampling，ICML 2025](https://proceedings.mlr.press/v267/de-bortoli25a.html)，§3、§8 | 正方差随机转移下，密度比接受与拒绝校正保持 target 分布 | 可解释我们为什么不能宣称 lossless：确定性 Euler 条件转移不满足该高斯转移前提。初始高斯噪声不改变这一点 |

FLASH 是另一个与机器人相关的参考，但本轮未核实其正式 A 会身份，仍按预印本处理，不作为用户要求的 A 会依据。

FlowCast 的正式论文可核验，但本轮未找到可核验的官方发布源码。后续应称“论文算法重实现及 WAM 适配”；在处理清楚原文 Algorithm 1 的状态/下标约定并完成结果对照前，不称已复现官方实现。

## 当前公式的数学含义

记同一状态和 conditioning 下的理想 Euler 更新为

\[
x'_{T}=x+\Delta\sigma_i v_T(x,\sigma_i),\qquad
x'_{D}=x+\Delta\sigma_i v_D(x,\sigma_i).
\]

则直接相减得到

\[
\|x'_D-x'_T\|_2=|\Delta\sigma_i|\,\|v_D-v_T\|_2=e_i.
\]

因此，乘 scheduler 步长有明确理由：把方向差换算成这一步引入的状态更新差。这是从本项目递推直接得到的恒等式，不是新定理；也不是数值分析中“离散解相对连续 ODE 真解”的局部截断误差。

当前有效向量为7通道×4帧×4动作的112个归一化latent分量，生产实现将 BF16 方向转 FP32 后计算 L2，未取 RMS；运行中的 BF16 舍入、条件帧投影及 packed verifier 与 canonical target 的数值差，使该分数不能直接当成最终提交状态差的逐位证书。应将其表述为原更新的一步缺陷代理。[生产 score](../../../投机采样/scripts/p0_batched_verify.py)。

与 FlowCast 常规均值型 MSE 的代数关系是

\[
e_i^2=D\,|\Delta\sigma_i|^2\operatorname{MSE}(v_D,v_T).
\]

在理想算术、`|Δσ_i|>0` 时，当前固定 L2 阈值等价于逐步 MSE 容差 `τ²/(D·|Δσ_i|²)`，不是固定 MSE 阈值；实际浮点实现仍需复核。在非均匀 scheduler 上，这个区别会真实改变接受位置。固定维度、固定比例的 L2→RMS 换算只是单位转换，不能作为方法创新。

## 0.007839680526450805 的可复核来源

独立 CPU 复算从保存的 Phase-B JSONL 边误差重建全部四个分位数，q50/q75/q90/q95 均与封存值逐 float 一致；没有重新执行 target 或把历史样本当新实验。

- 原始校准产物：2026-09-29 的 `p0-libero-long-phaseb-k2-b2-verifier-replay-20260929/phase_b_result.json`。
- 数据：历史 task8/init0 的 B1 离线两条边，共 1,200 anchors、2,400 个 successor-L2 值；task9 用于当时离线评估。
- 算法：NumPy float64、linear 第90百分位。排序位置2159.1；相邻值0.007839473581079776与0.007841543034790075插值，恰好得到当前阈值。
- 校准误差来自旧 B1 和历史精度流程；不是当前 A_off/BF16 在独立任务上通过成功率约束后选出的最优阈值。

这证明阈值有数据来源；它不证明90%成功率、不证明90%的独立轨迹质量覆盖，也不是 conformal/risk-control 保证。历史任务中的边互相相关，后续 draft 和访问状态分布也已变化。[复算及来源链](../../../review/OMISSIONS.md)。

## 能用于本项目的条件误差界

以下是针对固定 conditioning 的本项目递推推导，需自行说明假设；不是声称已经验证实际 WAM 全系统满足它。令 `F_i` 为同一离散 target 更新，顺序解 `x^T_{i+1}=F_i(x^T_i)`，候选实际缺陷 `r_i=hat{x}_{i+1}−F_i(hat{x}_i)`。若 `F_i` 在相关区域为 `λ_i`-Lipschitz，则

\[
\delta_{i+1}\le\lambda_i\delta_i+\|r_i\|,\qquad
\delta_N\le\delta_0\prod_{j=0}^{N-1}\lambda_j+
\sum_{i=0}^{N-1}\|r_i\|\prod_{j=i+1}^{N-1}\lambda_j.
\]

理想 Euler、同 conditioning、向量场为 `L_i`-Lipschitz 且同条件投影非扩张时，可取 `λ_i≤1+|Δσ_i|L_i`。浮点舍入与 verifier/原 target 差异需要额外项；实际有限样本测得的局部敏感度不能冒充全域常数。拒绝或 bonus 在理想同路径 target 更新下不增加新的方向替代缺陷，但不会把累计偏差清零。这个界解释了为何小的逐边偏差仍可能累积和放大，也解释了仅报接受率不够。

FlowCast 的相关推导提供思路，但应用前至少要厘清：原文 MSE 与范数的维度约定、均匀步长与本项目非均匀 sigma 的区别，以及其将全局接受比例 p 代入逐步偏差的步骤。对确定性接受排布，使用上面的逐步加权和更直接。论文中的 trajectory 在这里指去噪状态轨迹；用户关心的 trace 长度是完成任务所需的环境步数。连续 ODE 的误差预算不能直接推出后者或 LIBERO 成功率；后者还涉及反馈观察、接触和终止条件。

## 建议如何写论文、如何验证

当前方法可以如实写成：

> 我们采用步长加权的速度残差验证确定性动作流中的投机更新。该残差在理想 Euler 递推下等于同状态下一步更新的 L2 偏差，与 FlowCast 的速度一致性验证和 ParaTAA 的递推残差判据相关。接受最长合法前缀；现有阈值通过历史离线校准获得，任务质量另以配对闭环实验验证。

使用“approximate speculative inference / error-controlled verification”作为方法定位较准确。当前证据不足以使用“distribution-preserving”“lossless”或“guaranteed task-success preservation”。FlowCast 也使“连续 flow 中用速度差验证”本身不能再被当作本项目独有创新；学习型树候选、误差预算、阶段选择或系统实现是否构成贡献，需要各自消融和实测支撑。

下一阶段建议保持 PF2、target、draft 和运行条件固定，先比较三个接受规则，避免把周期、训练和阈值同时改动：

1. 保留当前步长加权 L2 作为历史基线。
2. 增加 FlowCast 风格的速度 MSE gate；另列原文恒速 draft 与当前学习型 draft，明确哪些是复现、哪些是适配。
3. 增加基于实际 scheduler 更新的残差控制，可参考 DPM-Solver 的绝对/相对容差归一化；使用原50次更新，不暗中换求解器。

新的容差应只在独立开发/校准 episode 上选择，按完整 chunk 输出、闭环成功与环境步数约束筛选，冻结后再做独立验收。比较相同成本或相近接受率下的质量曲线，不能用简单改单位的阈值重命名制造收益。达到质量门后再隔离测端到端耗时。大规模 LIBERO 前仍先完成此前约定的仿真提速与一致性核查。

本轮只完成文献、阈值来源与数学边界核查；上述新规则和基线尚未在 LingBot 上实测。

证据目录：[接受规则复核](../../../review/OMISSIONS.md)；[可引用 BibTeX](../../../review/OMISSIONS.md)；[历史阈值与数学审计](../../../review/OMISSIONS.md)；[三篇正式方法原文核查](../../../review/OMISSIONS.md)；[FlowCast与CCF目录独立核查](../../../review/OMISSIONS.md)；[DPM-Solver公式与固定作者源码](../../../review/OMISSIONS.md)。
