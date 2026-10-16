# 加速比测试口径：A 会文献与 LingbotVA 实验建议

调研时间：2026-10-13（Asia/Shanghai）。用户希望了解加速比如何测试，以及是否所有测试都必须运行 LIBERO 仿真。本文只引用会议正式论文、作者原文与 CCF 目录；不把文献中的模型推理速度改称完整任务加速。

**结论：速度测试可以主要使用真实输入与完整状态回放，LIBERO 闭环用于任务质量及真实任务耗时。两类结果同时报告。** 回放是针对本项目的建议；以下论文明确支持分开报告推理指标与任务质量，但不代表它们都明确采用了回放协议。

## 1. 相关正式论文

NeurIPS 与 ICML 都列于 [CCF 人工智能 A 类会议目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)。下列工作有主会议正式收录页，未将 workshop 或仅有预印本的工作冒充 A 会论文。

| 论文 | 会议 | 原文如何报告速度/质量 | 对本项目的意义 |
|---|---|---|---|
| [EfficientVLA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2ef31a1e45908901adc0ca853a8faf-Abstract-Conference.html) | NeurIPS 2025 | §4.1 在 A40 上测平均单次推理时长；Table 5 为 0.2342→0.1213 秒，1.93×。任务成功率另行报告，主实验是 SIMPLER。 | 加速比可以明确指一次完整模型推理，不必等同 episode 总时间。[正式 PDF，§4.1 / Table 5](https://proceedings.neurips.cc/paper_files/paper/2025/file/3a2ef31a1e45908901adc0ca853a8faf-Paper-Conference.pdf) |
| [VLA-Cache](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html) | NeurIPS 2025 | §5.1 分别报告 CUDA latency、control frequency、FLOPs、success rate。LIBERO Table 2 的 OpenVLA 延迟 51.91→31.83 ms（1.63×），频率 4.23→4.59 Hz（+8.5%），成功率 75.0→74.7%。 | GPU 延迟与控制层收益可能相差很大，必须保留各自分母。论文没有在这些段落充分说明频率与全部 env/client 时间的对应关系，不能自行断言。[正式 PDF，§5.1 / Table 2](https://proceedings.neurips.cc/paper_files/paper/2025/file/f062da1973ac9ac61fc6d44dd7fa309f-Paper-Conference.pdf) |
| [Accelerated Diffusion Models via Speculative Sampling](https://proceedings.mlr.press/v267/de-bortoli25a.html) | ICML 2025 | 主要比较 NFE 与质量；机器人实验为 PushT，报告目标调用数和 reward，质量评估运行 1000 episodes。 | 不限定 LIBERO；NFE 节省可作机制证据，但不能直接用作墙钟加速比。其随机采样与反射耦合的精确分布结论不能直接套用我们的 deterministic flow + direction threshold。[作者原文，§7](https://arxiv.org/html/2501.05370v2) |
| [Medusa](https://proceedings.mlr.press/v235/cai24b.html) | ICML 2024 | Appendix B.1 区分平均每轮接受/生成量、每轮额外延迟和墙钟 speedup；主要推理实验关注 batch=1，另有树规模及 batch/context 消融。 | “验证推进几步”要与“这一轮实际多贵”一起看。Appendix G 的简化块模拟省略部分后处理，不能把模拟曲线当我们完整系统的实测。[作者原文，Appendix B / G](https://arxiv.org/html/2401.10774) |

这些论文的计时实现细节不完全一致；没有从原文确认的预热次数、同步方法、统计重复数及是否包含每项预处理，不应补写成作者结论。论文标称倍数也不能横跨不同硬件、模型、任务直接排名。

## 2. 我们建议固定的四层指标

| 指标 | 计时范围 | 用途 |
|---|---|---|
| verifier/kernel 延迟 | 某次验证或共享注意力算子的明确范围 | 证明局部实现确实更快；不能代替投机算法加速 |
| 完整动作 diffusion 延迟 | 所有降噪步骤及 draft、验证、接受/拒绝、fallback、bonus、scheduler、主机调度 | 回答投机本身是否值得；50 步完整完成，不能只测一次 verification |
| 完整 policy/chunk 延迟 | 相同观察和真实历史状态下生成同样动作 horizon，包含视频/视觉处理、动作 diffusion、原输出拷贝及调用路径 | **建议作为日常主加速指标**；包含所有可重复的算法开销 |
| 完整 episode 墙钟 | reset、所有 policy/chunk、动作提交、通信和环境执行；从任务开始至标准成功/失败/horizon | 用户已要求的真实完整任务收益；保留，另报成功率与环境步数 |

prompt/T5、模型加载、模板首次构建的冷启动另行报告，并在实际发生的 episode 范围内如实计入；不能只把优化版的准备工作排除、却把 baseline 的相同准备计入。`reset` 目前每 episode 一次，不能摊成每次 diffusion 都执行。模板预热可用于稳态指标，不能掩盖首次请求成本。

模型采样目标调用次数、CFG 行数、接受深度和 committed scheduler updates 是辅助指标。批量验证的一次调用可能计算更多候选行；行数也不能直接等同 FLOPs。对固定 action horizon 的策略，action Hz 与 query Hz 分开定义；内部降噪步数不能充当机器人控制频率。

## 3. 如何减少 LIBERO 测试开销

日常迭代先做 CPU 语义和状态检查，再用已有真实固定快照做完整推理计时；只有通过精度和速度检查的候选才扩大闭环评估。固定快照必须包括真实观察、prompt/编码、有效历史 KV 与元数据、scheduler、RNG 和 streaming 状态。对跨 chunk 复用/共享缓存的实现，只重放一张图像或人工裁剪 KV 不足以代表真实运行。

未来正式速度评估应覆盖短/中/长 KV、首/后续 chunk、多任务与多初始状态。baseline T、上一版和候选版本用本轮相同 GPU、精度、输入状态、动作 horizon 和同步边界；保持一个部署模型、独占 GPU，按平衡/随机顺序重复。每次前后同步 CUDA 并用完整墙钟计时，计时外做状态恢复和结果核验；GPU event/profile 作为定位辅助，避免将异步 CPU 提交时间误报为完成延迟。

预热和主样本分开。建议后续发表级协议每个配置/条件收集至少 30 个有效测量，并进行至少三个独立运行；这是本项目的起始建议，不是论文统一规定。报告均值、中位数、p95、样本数、波动和按输入状态/独立运行聚类的置信区间，不能把同一快照的一百次重复当作一百个独立任务。现有 bonus pilot 是两个长 KV 状态、每种配置六次主测量，不能据此声称普遍加速或可靠尾延迟。

完整 LIBERO 闭环仍用于成功率、轨迹环境步数和 episode 墙钟。投机 threshold 或 bonus 会改变动作，观察也会跟着变；固定状态回放不能证明闭环成功率与最终轨迹质量。正式结果应覆盖选定 benchmark 的多任务、多 seeds/初始状态，遵守匹配的标准评估协议。两条试跑只作 pilot。发生早失败的 episode 可能更短，不能用“所有结束时间更短”直接认定任务执行更快；同时报告完成状态、成功任务耗时、失败/horizon 耗时、成功率和全体 episode 分布。

EGL 并行适合收集质量样本；同一卡上多模型会改变延迟，不能把其吞吐/并发 episode 时间当成单模型部署延迟。项目要求正式数据收集至少两个独立 EGL worker；现有固定快照回放不新增环境/episode，单模型 timing 与该要求不冲突。

## 4. 加速比和推进量的关系

对匹配输入、相同计时范围：`S = baseline 时间 / candidate 时间`。例如 10→8 秒是 1.25×，时间减少 20%；20% 时间减少不应写成 1.20×。多状态汇总需事先确定权重，可报告固定样本集总耗时比，同时给出每档/每状态结果；不只挑最好值。episode 要另外计算自己的耗时比，不能把各 kernel 的加速倍数相乘。

设平均每轮实际推进 A 个降噪更新，原 target 单步成本 t，投机每轮总成本为 draft + verify + host，则在忽略尾段及不变模块的近似下：

`S_action ≈ A × t / (t_draft + t_verify + t_host)`。

所以平均推进 1.75 步只是分子信息；批量 verifier 若比单步贵、或者 Python/同步调度较多，墙钟仍会不变甚至变慢。bonus 可在两层接受后利用叶子 target 方向再提交一次更新，但仍须实际测完整 50 步；不能按旧轨迹接受率推算的新平均推进量冒充实测收益。

按 baseline 对可优化部分的耗时占比 f，局部加速 S_local 的整体近似为 `1 / ((1-f) + f/S_local)`，只适用于工作量/轨迹没有改变的分解示意；闭环轨迹变化时需用真实 episode 计时判断。

## 5. 现有数据对“LIBERO 慢”的解释

最新 O3 两个完整任务总耗时 642.318 秒，其中 reset 280.727 秒（43.71%）、policy 299.832 秒、cache 20.241 秒、客户端余项 41.519 秒（6.46%）。余项是环境和客户端合计，尚未独立分离纯环境执行。详见 [已封存 O4 结果](OPTIMIZED-E2E-O4-RESULT-20261013.md) 和 [reset 分析](BASELINE-RESET-ANALYSIS-20261013.md)。

这说明我们当前不能把 episode 慢主要归因于纯 LIBERO 仿真。CPU T5 的 reset 是较大的固定成本，会稀释只优化 action diffusion 的收益。这不影响单 chunk/完整 diffusion 加速作为有意义的指标，也不应取消真实 episode 指标。

## 6. 当前 bonus 测试进展及下一步

已有 bonus 分支复用现有最佳实现。同一共享 CPU 注意力后端下，原 helper 与 O3 helper 六组完整 BF16/50 步/30 层输出、有效 KV、scheduler、RNG 对照通过。原始未共享 CPU 注意力对照仍失败：首组 46 个输出元素不同，最大绝对差 0.015625，分支计数一致；该结果完整保存，不放宽逐位门槛、不称原 Torch 精度已通过。后端变化是排查方向，CPU 数据本身尚不能代替真实 GPU 结论。

GPU 后续先跑原始未共享 Torch bonus 与优化 bonus 的真实长 KV 完整 chunk oracle，再测 T/OFF/BONUS 三方完整推理，最后按已授权方案跑六个完整 LIBERO episode。准备阶段采样显存/进程为空但利用率约 25%，未提前启动计时；随后连续 15 秒确认利用率 0%、进程/分配显存为空，已通过严格准入，先启动一个模型进行 oracle 对照。本文写入时 GPU 对照尚在进行，不预告速度或质量通过。本轮 finite plan 最多新增 16600 次 target 前向、3600 秒 GPU，累计仍低于用户 5 小时上限；没有第三层 draft 或 Megatransformer 实现。

结果与资源状态以 [bonus 独立方案](../../../投机采样/experiments/bonus_e2e_v1/PLAN.md) 和 evidence/job 的最终记录为准。后续优先完成这个 bonus 消融，随后把固定状态完整推理作为短/中/长 KV 的常规速度回归；任务质量仍按 LIBERO 闭环验收。
