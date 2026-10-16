# Realtime-VLA FLASH / Fast-WAM：结构、投机范围与发表路线

2026-10-11T22:38:25+08:00（Asia/Shanghai）；投机采样主线，**非tile算子研究**。

## 1. 本次问题与判断

用户先问FastVLA/FastWAM是否使当前投机失去意义，后明确前者是 **Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs**，并询问面向CCF A是否应覆盖更常见的整块动作范式。本次仅文献/本地源码读取，不新增GPU/训练/服务，不改既有执行合同。

**判断：投机没有因action chunking失效，但只在第一代LingBot重50步路径上获得局部加速，不足以说明可迁移性或强发表竞争力。** 以下“主流”仅指有代表性的公开设计，不是基于市场/使用份额的统计排名；Fast-WAM是合适候选而非已证明唯一主流。建议保留现有证据/先成本关，再扩大问题验证范围；不是现在授权重构。

## 2. 必须区分的三条轴

- H：每块包含多少未来物理动作，及每次实际执行多少再取新观测。
- S：生成同一块动作要多少次flow/扩散更新，整块动作可以在每次更新中并行。
- B：同一target调用并行多少候选/CFG行；forward减少不等于rows/FLOPs/访存或wall time减少。

当前LingBot已经同时预测16个未来动作，内部是50action scheduler updates加terminal cache forward；并非a1、a2逐个自回归吐出。K2B2预测的是后续去噪direction/transition，7逻辑/14CFG候选由真实target验证，global阈值仍为原值。这里的accepted depth是去噪更新数，不是环境动作数；外层闭环chunk历史是另一回事。依据：[冻结server](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)、[当前固定合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)与实际GPU G1/S1账本；未改它们。

| 范式/证据 | 动作生成结构 | 对当前去噪投机的含义 |
|---|---|---|
| 当前LingBot公开shared-backbone路径 | 未来video+action分支，缓存长历史；整块16动作反复更新 | 已有方法适用范围，不代表整个MoT发布版本或2.0 |
| [π0/π0.5官方实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py) | VLM前缀一次、action expert flow循环，sample_actions默认num_steps=10 | 概念上可研究，但cache/CFG/步长/draft输入/验收须重新对齐 |
| [π0.7论文](https://arxiv.org/html/2604.15483v1) | 50动作块，论文用5次去噪、执行15/25个再更新 | 已存在更少步的强结构；不能只拿50步作baseline |
| [Fast-WAM](https://arxiv.org/html/2603.16666v2) | 视觉single-pass，1B action expert/32动作/10次flow去噪 | 不显式生成未来video，仍有动作去噪链；收益空间须实测 |
| [OpenVLA-OFT](https://openvla-oft.github.io/) | 并行chunk、连续动作、L1回归 | 没有同样的多步去噪链，当前transition投机不能原样迁移 |

## 3. FLASH：最近先行工作，不是同名视觉压缩方法

[论文](https://arxiv.org/html/2605.13778v1)、[项目](https://dexmal.github.io/realtime-vla-flash/)和[官方源码](https://github.com/dexmal/realtime-vla-flash)对应同一研究。其π0 full path含image encoding、VLM prefill、10-step动作去噪；fast path用当前图像编码+单Gemma block draft整块动作，复用上次full path的prefix KV。target Action Expert在选定noise/action插值状态上并行预测velocity/端点，以动作维度距离得到保守可执行前缀；夹爪切换等精细阶段回退full path。

其优势包含跳过VLM prefill和多步动作解算，并不只是少算两次target。作者表1中Torch π0→FLASH为58.0→34.9ms（1.66x）；FLASH+Triton为19.1ms（相对Torch3.04x，相对39.7ms Triton baseline约2.08x），平均success94.1→93.8%。7.8ms仅fast-path round，**不是整体episode wall time或动作质量精确等价**。[官方项目](https://dexmal.github.io/realtime-vla-flash/)

与我们相比：FLASH draft最终物理动作/验证可执行时间前缀；我们draft内部去噪transition/验证sigma更新前缀。对象、条件上下文和缓存策略不同，不能直接替换验证器或比较原始毫秒数字。论文附录B明确是局部一致性启发式，不保证同full-path轨迹/任务行为相同；需闭环测success/steps/tail。[论文附录B](https://arxiv.org/html/2605.13778v1)

可借鉴（建议、未实现）：

1. **执行相关监督**：[官方训练源码](https://github.com/dexmal/realtime-vla-flash/blob/main/scripts/spec/spec_draft_train.py)采用有时间/前缀权重的Huber，优先关心即将执行的动作，而非全latent平均L2。对我们需从transition偏差联系到最终执行动作/任务关键维，不能简单照抄endpoint loss到不同预测对象。
2. **任务敏感回退**：精细抓取/释放阶段保守target，这与旧LIBERO夹爪闭合后contact返工的现象相符，但尚非因果证明。只加夹爪回退不是新意，且未授权改当前验收/阈值。
3. **现代成本基线**：必须和已优化AE/少步推理相比，分开算法与kernel/compile收益，不将FLASH+Triton全部收益归算法。
4. **未解决的问题才是候选贡献**：控制任务步数/失败长尾与成本预算，候选生成/verification是否能随负载及任务风险动态取舍。FLASH已含phase-aware fallback，不能宣称首次。

## 4. Fast-WAM：single-pass视觉不是single-step动作

[Fast-WAM方法](https://arxiv.org/html/2603.16666v2)通过训练attention mask禁止action访问未来video、共享clean first-frame表示；保留video联合训练信号，但推理只编码当前视觉表示一次，仍做10步动作去噪。**不能对冻结LingBot直接删video分支并继承该结果**：上下文信息流/训练条件不同。

当前[官方仓库](https://github.com/yuantianyuan01/FastWAM)还提供compile加速与Optional IDM同权重两模式，作者报告LIBERO平均IDM98.55%/first-frame97.75%，表明省未来并非所有条件无质量代价；这是它自身设置，不是本项目结果。后续[Faster-WAM](https://arxiv.org/html/2608.04404v1)在OOD上研究保留一次未来表征再缓存/稀疏交互，进一步提醒：视频不总是无用，收益需要以分布/任务来界定。当前两篇的公开版本为预印本/官方代码，未核实其CCF A录用，不靠“未中稿”判断方法好坏。

## 5. 面向CCF A的研究建议（不是录用保证）

我不认为问题只是LingBot“不主流”。同名工作已在[RSS2026正式论文集](https://roboticsproceedings.org/rss22/p016.html)出现；该正式摘要称CauVA且实现设计有变化，不能把它直接当我们冻结shared-backbone版本的全部性能保证。

已有[ICML2025 Accelerated Diffusion Models via Speculative Sampling](https://proceedings.mlr.press/v267/de-bortoli25a.html)研究连续vector-valued Markov diffusion的精确投机采样；[CCF ICML条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/al/2017-04-25/592030.shtml)列A类。它说明连续输出并不排斥投机，但其随机transition/密度与exact保证不自动适用于我们的确定性flow+L2近似验收。

研究问题建议收束为：**在多步连续action-chunk生成中，能否通过draft/verification的训练与调度，获得闭环任务非劣、环境步数/长尾受控且真实净成本降低的可迁移方法？**

- 结构覆盖候选：π类轻action expert + Fast-WAM action-only + LingBot长历史压力案例；优先真实能复现的公开权重，不要求立即全做。
- 比较候选：已优化原模型、5/10步等质量匹配few-step、FLASH、阶段/负载保守回退；有些无法实现/不适用要明示，不拿未优化50步作为唯一强基线。
- 本项目现有init3/4固定chunk与单任务小pilot不足以支撑跨模型/跨任务、success非劣或发表强结论；每次正式任务评测需预注册容差、失败处理、paired seeds/initial states和独立episode样本数。
- 方法贡献不能只是“换Fast-WAM”“加RL”“加KL”或“加夹爪fallback”；应明确相对FLASH/连续diffusion投机的新机制及消融，且正负成本都报。
- 指标至少分开success、env steps、median/长尾、总policy/GPU work与episode E2E；[最新评测审计预印本](https://arxiv.org/html/2609.37771v1)指出基准checker/初态/复现缺陷可能改变加速排名，尚不能用它断言本项目benchmark有bug。

以上是研究竞争力判断/候选实验清单，不是保证可发CCF A，也不等于已获迁移、RL/联训或额外GPU授权。

## 6. 当前执行状态与下一步

记录时本次固定负载job仍为 main_L1_slot83，已过 INITIAL/FIXTURES/G1/S1/L2；固定chunk已预留/消耗 124/168，EGL 2/2，完整episode0。正式成本只按已通过关卡引用，不把未验证raw timing算结论。文献工作新增GPU=0、训练/RL/联训=0。

先完成本次已授权固定成本GPU矩阵/终审与清理，不改当前模型、阈值或预算，不训练。随后仅CPU汇总成本/工作账本并制定跨范式可迁移性与新意检查：Fast-WAM/π类5–10步动作专家、FLASH对照、任务成功/环境步数/长尾/E2E联合合同。迁移实现、新GPU或训练均另确认，不因讨论自动重构。

[当前job](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036)；[投机研究总入口](RESEARCH.md)。
