# MegaTransformer与tile方向整合决策

> 最新夜间结果（2026-10-15T05:45:28+08:00）：45次训练无稳定新权重，保留A_off；30条开发质量任务、新TRAIN与PF2/3/4已独审。算子融合已撤回，原tile不改。见[统一报告](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[扩大准备](NEXT-SCALE-READINESS-20261015.md)，HW记录 CHAT-20261015-OVERNIGHT-DRAFT-ACCEPTANCE-RESULT。


> **2026-10-14 研究追加（CHAT-20261014-DRAFT-DATA-VERIFIER-FOLLOWUP）**：历史Long4802行重建、15组CPU训练、6份真实profile、64个CUDA数值案例及6个完整chunk验收已完成；新draft未通过多seed原准入，保留A_off。四处融合数值一致，正式性能待独占；42chunk私有导入入口已封存。累计GPU31065.8421秒/147346 intents，余82.24分钟，自有训练/GPU已退出。详细结果与下一步见 [本轮报告](DRAFT-DATA-VERIFIER-RESULT-20261014.md)。以下原报告与历史状态保留。


> 最新结果（2026-10-14T20:10:22.540+08:00）：[18组draft结构初筛与verifier状态](DRAFT-ARCHITECTURE-RESULT-20261014.md)：Transformer在6/6组公平配对中优于容量相近MLP，但全部候选未过准入，保留A_off；低学习率root中位515/630对527/630，退化集中中sigma root。先核验旧train来源并扩大有效数据。verifier profile代码/CPU预检完成，GPU外来占用未启动，尚无融合原型或新性能收益。本轮0新GPU、余90.83分钟，自有训练/监测已结束；tile及preset保留。Knowledge：CHAT-20261014-DRAFT-ARCHITECTURE-SCREEN-RESULT。下方“尚未训练”等为历史时点。

> 最新安排（2026-10-14T19:16:40+08:00）：[draft选型与verifier并行计划](DRAFT-STRUCTURE-PARALLEL-PLAN-20261014.md)：先冻结接口，有限对照三类小候选，筛选首版结构后扩大训练分区；系统线可同时profile当前CFG14并做隔离block原型。现有训练便宜，优先补数据覆盖与闭环证据；旧Transformer shadow不能选定当前结构。已完成静态复核，0新GPU，尚未启动新训练/profile；余GPU90.83分钟。tile和当前preset不改。Knowledge：CHAT-20261014-DRAFT-STRUCTURE-PARALLEL-PLAN。

> 最新完成（2026-10-14T18:53:35.594+08:00）：[三档完整policy chunk实测](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)：短/中/长KV加速1.602×/1.544×/1.500×，45完整chunk、18/18正式配对胜出，同算法现场数值/KV/RNG及独立CPU审计通过。两次失败保留，最终累计GPU30550.488秒/36000秒（余90.83分钟），前向收费144550；自有进程清理完成。此前“待办/本次未测得”仅为当时状态，以本结果为准。[MegaTransformer整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已固定范围，下一步profile和单block原型；tile代码/HW不改，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-LENGTHS-COMPLETED。

2026-10-14，Asia/Shanghai。用户授权恢复完整chunk测试，并参考tile研究固定MegaTransformer方向。本页是工程范围与项目组织决定，不是论文初稿，也不声明新kernel已实现。tile方向本轮由subagent只读核查，未使用GPU、导入Torch、运行测试或修改文件。

## 结论与统一目标

可以作为同一个研究项目推进：**在任务质量约束下，加速WAM的投机动作生成。** 算法线减少串行target调用并控制接受/回退；MegaTransformer系统线降低每次候选验证成本；最终用两者组合的完整chunk和闭环任务检验收益。

研究目标与论文证据合并，现有源码暂不物理合并。tile保留为已验收的底层基线和实现参考，在新隔离目录推进CFG14 verifier实现；待出现经过验证的公共接口再抽库。不能把现有tile交付描述成已经完成的MegaTransformer，也不能把不同基线的百分比相乘。

## tile真实进展和可用价值

有效最新结果以[tile PyTorch2.9最终报告](../tile-sdpa/PYTORCH29_FINAL_20261012.md)为准：固定当前PyTorch对应FlashAttention/CUTLASS源码，独立namespace编译，不回落到内置SDPA。两进程、六真实fixture、七轮测试，576项正确性、48/48 attention、12/12完整block门通过。

| 层级 | 最新相对PyTorch中位耗时下降 | 解释范围 |
|---|---:|---|
| eager attention | 17.31%–49.06% | 包含主机投递，不能直接当GPU计算收益 |
| CUDA core | 0.36%–8.59% | 固定fixture的attention核 |
| 完整单block | 0.1060%–4.6062% | 尚未证明完整30层/任务加速 |

旧2.6版本9%–20%的core收益不能替代最新2.9结果。长video block最新仅约0.1%–1.7%收益；历史负结果保留。tile的数值门为atol=rtol=0.02，并非完整生成逐位一致；原PyTorch成功episode不能当candidate成功率。

## 接口兼容与不应直接替换的原因

两条实现钉住相同PyTorch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`及FlashAttention commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`，针对SM80/BF16/head_dim128。具体commit以源码冻结清单为准。

| 项目 | tile已测实现 | 当前投机共享路径 |
|---|---|---|
| query | B2，Q16/Q128 | `[14,16,24,128]` |
| history KV | dense，同Q batch | CFG2历史池＋14行私有当前KV |
| 索引 | contiguous stride | valid slots＋CFG row map |
| 调度 | standard/split-KV启发式 | B14×H24=336 CTA，split=1 |
| 数值验收 | 局部allclose | 已有完整verifier及chunk同算法精确对照 |

直接调用tile的dense接口需要重新复制/拼接14行历史，会破坏已有共享优势。B2的split-KV收益也不能外推到B14。当前共享verifier已有5.170557→3.359750秒（−35.02%）和额外峰值891.035→257.085MiB的实测，它是MegaTransformer的直接起点。

可复用的是版本/许可证管理、native入口和stream/device guard、真实fixture与分层计时、逐阶段block oracle、失败诊断经验。不是照搬dense kernel作为新的投机方案。

## 固定首版范围：CFG14 verifier的block内融合与执行复用

第一阶段只优化当前固定工作负载。保留target数学、接受规则、draft、50次action更新、共享history语义，以及现有attention、GEMM、LayerNorm/RMSNorm reduction和FP64 RoPE计算。

1. 用当前preset的真实CFG14短/中/长fixture，独立profile区分linear、attention、逐元素、cache和主机提交；profile与正式速度分开。
2. 若小算子/中间张量确有成本，融合相邻FP32 modulation、gated residual和dtype转换，使用固定工作区；保留原BF16物化与舍入边界，审查FMA contraction。
3. 新执行器先覆盖一个真实block，逐阶段验证后再推广到完整30层verify。
4. 若主机投递是剩余瓶颈，固定形状执行图作为独立消融；不能将单纯CUDA Graph包装写成自研Transformer megakernel。

该方向借鉴用户提出的MegaMoE式组织思想，但首版不承诺一核融合30层。跨CTA同步、寄存器/共享内存驻留和GEMM调度尚未解决。多候选共同加载同CFG历史tile作为后续研究候选；直接合并候选Q可能将336 CTA降到48 CTA，不能默认更快。

若profile表明上述小算子不是主要成本，应据证据调整实现目标，不为了保留名称而继续实现无收益融合。

## 验收与并行安排

验收顺序：真实block逐阶段数值 → 完整30层verify输出与history/slots状态 → 同算法完整chunk动作、接受/拒绝/bonus路径、cache、RNG → 无争用短/中/长完整chunk计时 → 组合闭环质量/任务耗时。

首选保持工程对照逐位及路径一致。若达不到，应先定位、修正或关闭；不能仅凭tile局部0.02误差门把新实现视为无扰动，也不能省略闭环质量。未达到完整chunk收益之前，micro/block收益只能按对应层级报告。

算法质量线可并行处理已有数据和接受/刷新方案，系统线使用冻结当前preset开展。单卡GPU正式计时串行执行，文字、静态分析可并行。算法候选和系统实现各自通过后再做组合及单独消融；不能同时改变两者而无法定位收益与质量退步。

沿用目标：10/17协议/候选收口，10/21更广质量，10/26一个系统原型数值与完整推理验收，10/30主证据冻结，11/01 18:00给老师完整稿。新GPU预算有限，以上是交付目标而非保证；若10/26无增量收益，MegaTransformer退出本次主结果承诺，保留现有共享KV系统证据。

## 源码证据入口

- [tile独立CUDA入口](../../../review/OMISSIONS.md)
- [当前共享verifier接口](../../../投机采样/experiments/shared_prefix_torch_v2/csrc/shared_torch.cu)
- [tile逐阶段block数学边界](../../../review/OMISSIONS.md)
- [当前模型block与modulation/residual](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py)
- [当前共享verifier实测](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)

本轮不移动/删除/重写tile代码、HW或历史结果，不改当前preset，不生成本地论文正文。
