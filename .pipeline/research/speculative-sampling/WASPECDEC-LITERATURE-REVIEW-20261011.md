# WA-SpecDec 与相邻文献：投机采样方案复盘及取舍

记录时间：2026-10-11T17:32:50+08:00（Asia/Shanghai）  
范围：LingBot-VA 连续动作去噪的投机采样；**不是 tile 算子研究**。  
状态：文献调研和 HW 记录完成；下述方案调整是建议，**尚未成为用户批准的代码、训练或 GPU 任务**。

## 0. 先给结论

**继续现有主线，不现在推倒重构；升级研究假设，而不是照搬整套 WA-SpecDec。**

1. 保留冻结原 target、固定验收规则、可复现闭环及成本账本。
2. 先完成已规划、尚待批准的“短/长历史 × 单/双需求”固定负载成本关。
3. 如果存在可回本的区域，再测试：**给 draft 补充已有的世界/场景条件，结合当前候选实际访问状态进行监督训练**。
4. 接受长度、L2 和 target 调用数均不能单独作为目标。最终目标是：**成功率不明显下降、环境步数及失败长尾受控、完整任务实际耗时下降**。
5. RL、改变 target、大小模型权重联训及新执行步长暂不进入当前关卡。

用户附件是研究提问清单，并未附网页端助手的完整回答；本报告不把其中的疑问当成已证实事实。

## 1. 项目检查点：已经做了什么，现在在哪里

调研前已单独保存 [原进展与未来关卡检查点](RESEARCH-CHECKPOINT-BEFORE-WASPECDEC-20261011.md)。核心事实如下：

| 阶段 | 已取得的结果 | 不能过度解释为 |
|---|---|---|
| 推理机制 | action 分支 draft→真实 target 批量验证→接受/回滚打通；历史稳态单 chunk policy 6.277→4.656 秒 | 所有 KV/负载/任务都加速 |
| 任务行为诊断 | 曾出现环境步数增加，单 chunk 改善没有稳定转化为任务总耗时改善；发现第二罐抓取返工等阶段性差异 | L2 是唯一原因；所有初态都变差 |
| 最小冻结 target 对照 | 20 局完成；同预算 A_off/B_on 更新。A_off 4/4 成功、总步比 T 少 27，但平均 E2E 慢约 9.67%；B_on 3/4 成功 | 真实状态训练普遍无效；B_on 是 RL 或多轮候选自身 on-policy |
| 可复现与观察器门 | 种子/状态/RNG/RGB 指纹补齐；固定动作及完整 OFF/OFF/ON 回放通过 | 所有未记录的内部 hidden 都已证明完全一致 |
| 最新六局 T/A_off 不扰动 | 全成功；T 293/300 步，A_off 283/291 步；ON 精确匹配各自 OFF 的完整 action/video 与已测轨迹 | 跨任务非劣、干净自然 E2E 已通过 |
| CPU 成本账本 | 32 对匹配请求，覆盖 64/75；action 平均 +25.93%，短 KV −10.14%，长 KV2160 +82.83% | 32 个独立任务实验；KV 长度已被唯一因果隔离 |
| 当前停点 | 固定负载最小方案已 CPU 规划，**新重放包装尚未实现，预算待确认** | 新工程 CPU 或 GPU fixture 关已通过 |

最新两初态里环境步数反而少 19、chunk 总数只少 1，仍有明显 action 成本退化。因此现在必须区分两个问题：

- **物理行为问题**：抓取返工、掉落、恢复动作、失败长尾。
- **计算回本问题**：候选验证的批量/历史/并发成本是否超过节省的串行 target 计算。

证据：[成本 CPU 读数](COST-LEDGER-CPU-READOUT-20261011.md)、[六局结果](GPU-NONPERTURB-SIX-RESULT-20261011.md)、[最小冻结对照](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)。

## 2. WA-SpecDec：事实与效果口径

正式名称是 **WA-SpecDec: World-Aware Speculative Decoding for Vision-Language-Action Models**。截至本次检索，arXiv 标为 preprint，未核验到正式接收；这不等于“已被拒稿”。它测了 LIBERO 四套任务和 SIMPLER-Env，不只是一个模拟基准；未找到真实机器人实验。[摘要与发表元数据](https://arxiv.org/abs/2608.08725)

其要点：当前图像经世界表征形成 bias，加入原 visual embeddings、供 target/draft 共用；用动作监督和下一帧 latent 辅助损失训练 WAB及 target LoRA，再冻结新 target 蒸馏 draft。推理不生成未来视频，原宽松验收规则不变。[方法 §3](https://arxiv.org/html/2608.08725v1#S3)

最能解释收益的是固定 rho=10 的消融，而不只看摘要：[原文表6](https://arxiv.org/html/2608.08725v1#S4)

| 方法 | 成功率 | 相对 AR 的解码加速 |
|---|---:|---:|
| AR | 72.2% | 1.00× |
| WA target-only | 73.5% | 0.97× |
| Spec-VLA | 65.4% | 1.20× |
| WA 无 video loss | 69.4% | 1.31× |
| WA 完整 | 71.0% | 1.34× |

**我们据此计算**：同配置增量为 1.34/1.20−1≈11.7%；video loss 相对无该项的增量约 2.3%。摘要“同成功率下 1.5×”是另一工作点口径，不能当成每个固定配置、每次任务均获 50% 增益。

### 我的解释，而非论文已证明的因果结论

- 这不是“世界模型预测更多视频所以免费变快”。更合理的解释是：给小模型及参考策略提供更有用的条件，减少难状态中的错误候选，让并行验证更值得做。
- 从消融看，表征/条件改善值得优先研究，辅助预测是进一步增益；不能把全部提升归给一个 MSE 项，也不能据此证明学到了完整物理规律。
- target-only 没有加速；**改善行为并不自动改善计算**。这与我们最新步数减少、action 却变慢的事实相呼应。
- 它的 AR 动作 token 与我们跨历史的连续去噪树不是同一成本结构。其加速数字不能直接替代我们的完整 episode 实测。
- 原文有 target-only 消融，不能指责其完全没有分离 target 改善；但我们要回答的是“冻结原 target 是否能更快且不损质量”，照搬 LoRA 会换掉这个问题。

### 指标及理论边界

原文速度计入 draft/验证/WAB/cache，但没有给出我们需要的环境步数和完整任务 E2E 对照；NCF 的分母是失败回合，条件是回合中曾接近物体/容器。[指标 §4.1](https://arxiv.org/html/2608.08725v1#S4.SS1)

据此应另外检查失败原因、接触阶段暴露量和首个偏离时刻；不能把 NCF 直接叫“碰撞概率”。失败集合和接触次数变化都会影响条件指标。

世界感知、宽松距离门与严格分布保持也要分开。下面的反例不依赖任何特定论文：target 确定输出 100，draft 输出 102，距离容忍为 3。如果接受 102，输出就不再服从原 target 的点质量分布。改进场景条件不消除这个反例。因此宽松模式应宣称**近似任务质量权衡**，不能仅因有 target 验证就称 lossless。

## 3. 相似文献：分类、发表核验与阅读顺序

### 分类口径

- **A：严格投机采样**——正确的条件概率/转移密度、接受规则与拒绝校正共同保持所定义 target 分布；贪心精确解码是另一确定性特例。
- **B：近似投机**——距离、运动学、经验一致性门；任务性能需另验证。
- **C：相邻系统优化**——缓存、压缩、异步预计算、跳过验证等，不能冒充严格采样。
- **D：执行层验证/训练**——判断是否继续执行或重新规划，或 RL 训练策略；不是 draft–target 概率校正。

“同一论文包含 A/B/C 多个模式”时必须按实际模式标注。

CCF 采用 **2026 年第七版正式目录**：ICML、NeurIPS、ICLR、ASPLOS、DAC 属 A；EMNLP 属 B。尤其 ICLR 的历史年份与当前目录认定不要混为一谈；学校认定按其实际规则。目录只针对正式 Full/Regular 主会论文。[CCF 目录入口](https://www.ccf.org.cn/Academic_Evaluation/By_category/)；[高校保存的 CCF 正式目录，AI pp.57–58/体系结构 p.5](https://kyc.lsu.edu.cn/_upload/article/files/20/77/2cbaa3754eb9aff9ed74cafed8ff/23a8de02-594c-445f-b084-69b0193c05b3.pdf)

### 已有正式论文集，可优先读

| 优先方向 | 论文 / 正式发表 / 当前 CCF | 类型及与我们直接有关的点 |
|---|---|---|
| 训练，最优先 | [DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)，ICLR 2024，A | A 主模式，另有 lossy 模式；draft 生成数据和匹配任务/解码方式的蒸馏，比只在固定离线数据拟合更值得借鉴 |
| 成本与 RL 边界 | [Learning To Draft](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)，ICLR 2026，A | 调度深度和验证候选数；优化真实接受收益/实际耗时。训练的是两个调度策略，不是大小基础模型权重联训 |
| 最接近连续动作机制 | [Accelerated Diffusion Models via Speculative Sampling](https://proceedings.mlr.press/v267/de-bortoli25a.html)，ICML 2025，A | A：随机 Markov 转移密度、并行验证和拒绝校正；直接适用性受采样器形式限制，不能把确定性 flow/L2 门叫成其定理的实例 |
| 在线数据 | [Online Speculative Decoding](https://proceedings.mlr.press/v235/liu24y.html)，ICML 2024，A | 固定 target 下随实时查询更新 draft；在线蒸馏不等于机器人 RL |
| 最直接 VLA 参照 | [Spec-VLA](https://aclanthology.org/2025.emnlp-main.1367/)，EMNLP 2025 主会，B | B 宽松动作 token 验收；AR/树验证、离散动作及解码收益的直接基线 |
| 场景条件的另一来源 | [EAGLE](https://proceedings.mlr.press/v235/li24bt.html)，ICML 2024，A | A：利用 target 高层特征 drafting；支持先复用已有特征，不一定新增完整世界编码器 |
| 树验证系统基线 | [SpecInfer](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html)，ASPLOS 2024，A；[DOI](https://doi.org/10.1145/3620666.3651335) | A：树结构并行验证；需要理解 mask、内存和 serving 条件，不能仅比较减少的调用次数 |
| 严格理论起点 | [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)，ICML 2023，A | A：概率接受和 residual 拒绝采样，而不是阈值距离 |
| 扩散结构补充 | [Diffusion Models are Secretly Exchangeable](https://proceedings.mlr.press/v267/hu25d.html)，ICML 2025，A | DDPM auto speculation 的结构性并行化；不能未经证明外推到当前 flow ODE |
| 非投机加速参照 | [EfficientVLA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2ef31a1e45908901adc0ca853a8faf-Abstract-Conference.html)，NeurIPS 2025，A | C：层/视觉 token 压缩与扩散缓存，训练免费不意味着无行为差异 |
| 缓存参照 | [VLA-Cache](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html)，NeurIPS 2025，A | C：相邻图像变化和任务重要性驱动缓存；不是 draft 接受/回滚 |
| 后置 RL 基础 | [Diffusion Policy Policy Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html)，ICLR 2025，A | D：机器人 diffusion policy RL；不是已证明可直接套到本 flow draft，亦非 WA-SpecDec 的训练方式 |
| 最新在线训练补充 | [Aurora](https://proceedings.mlr.press/v306/wang26j.html)，ICML 2026，A | 在线 target 验证反馈训练 draft、训练/服务协同；反馈奖励不等于真实环境任务奖励 |
| 并发与漂移补充 | [FastGRPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/61674667d642ae52f6bb281bea90ee29-Abstract-Conference.html)，ICLR 2026，A | 并发感知投机和在线 draft；其 E2E 是 LLM RL 训练口径，不是机器人完整任务耗时 |

优先读前五篇，其余用于机制和边界补充。没有必要为凑 A/B 数量把每一篇都实施。

### 附件提到、但需区别发表状态的工作

| 工作 | 截至检索的发表证据 | 不能混淆的机制 |
|---|---|---|
| [KERV](https://arxiv.org/abs/2603.01581) | 作者 arXiv v2 与[代码仓](https://github.com/zhengzihaoPKU/KERV)报告 DAC 2026 接收（当前 A）；未独立核验正式 DOI | B：运动学补偿/动态容忍；不是严格分布保持。仓库后来新增的系统优化不要全部算原论文贡献 |
| [HeiSD](https://arxiv.org/abs/2603.17573) | 未核验正式接收 | B+C：生成/检索混合及 verify-skip；跳过验证的长度不能与真实验证接受长度混算 |
| [DexWorldModel/CLWM](https://arxiv.org/abs/2604.16484) | 未核验正式接收 | C：未来条件下先去噪，再用真实观测更新完成；主要是异步/阻塞延迟，不是 p/q 拒绝采样 |
| [When to Trust Imagination / FFDC-WAM](https://arxiv.org/abs/2605.06222) | 未核验正式接收 | D：执行中判断继续或重规划；FFDC 指 Future Forward Dynamics Causal Attention，不是附件猜测的名称 |

WA 的模拟范围、KERV 的作者接收声明、Spec-VLA 的正式主会发表，均与初始猜测有区别。不能按“预印本=不好”“方法简单=无效”筛选，也不能把作者接收声明和已核验正式论文集混为一谈。

## 4. 把 AR token、连续去噪和环境执行三层分开

### 4.1 通用的前缀验证语义：以下是我们的示意，不是原文代码

```text
context = observation + instruction + committed_prefix
draft = propose(context, K)
teacher = verify_in_parallel(context, draft, correct_causal_mask)
for i in draft order:
    if reject(draft[i], teacher[i]):
        commit a valid target/corrected replacement for this prefix
        discard every later draft item and its speculative cache
        break
    commit draft[i]
```

因果 teacher forcing 验证第 i 项时只能看到其前缀，不能看到答案本身；树分支也只能看到祖先。若草稿为 100/120/130/140，对应参考为 101/119/200/141，容忍 5，则接受前两项、第三项拒绝；第四项即使局部“接近”，其计算依赖了已无效的第三项，不能继续提交。

这是 AR 示意。我们的第 j 个去噪候选是某个带时间/条件/cache 的**状态转移查询**，不是第 j 个真实环境控制时刻；将其一一等同会误解步数和风险。

### 4.2 精确采样与近似门

标准离散采样在同一前缀下用 α=min(1,p(y)/q(y)) 接受；拒绝时从 r(y)∝max(p(y)−q(y),0) 采样校正。只有接受比、拒绝校正、条件前缀等全部正确，才有严格保证。[ICML 2023 原论文](https://proceedings.mlr.press/v202/leviathan23a.html)

连续随机扩散需要转移密度及相应耦合/校正；原 ICML 2025 方法明确区分随机链与确定性采样器。它的 NFE 可按一整次 batch 计一次，不能替代 batch 行数或墙钟成本。[连续扩散原文 §8](https://arxiv.org/html/2501.05370v2)

当前确定性 residual direction + 连续 scheduler 的距离门属于 B 型近似验证。若以后研究严格随机分支，必须另定义 target 随机链；注入噪声后的链不是“原确定性 target 的免费精确化”。

token 差 5 也不能直接换成某个厘米数：要知道动作归一化范围、维度、bin 边界、反量化及执行控制器。连续 L2 同样不能自动映射为抓取成功或接触风险。

### 4.3 接受率为什么不等于速度、KL 为什么不等于步数

令 L 为每轮接受的草稿前缀长度，则一般有：

\[E[L]=\sum_{j=1}^{K}P(A_1\cap\cdots\cap A_j).\]

只有独立且同概率等强假设下，才可简写为 Σα^j；不同深度/物理阶段明显不应默认独立。是否额外提交 target 修正或 bonus，取决于协议，不能把别的论文的“L+1”直接套到当前 K2。

在重复负载和近似稳定成本等条件下，可用下式作预算模型，而非任务性能定理：

\[S_{action}\approx\frac{E[m]\,c_T}{c_{draft}+c_{verify}+c_{restore}+c_{other}},\]

其中 m 为实际提交的 target 更新数；所有候选行、拒绝、构树/调度、offload 和等待都计费。然后还要加入 video、环境执行、观测/通信和任务返工；完整任务速度无法由单项 S 推出。

我们已有成本证据：50 次 action 更新、K2 最多两次提交时至少需 25 次验证；**若长段每次验证仍为约 .401 秒**，仅此就约 10.03 秒，高于该段 T action 约 8.235 秒，尚未计 draft。这是有条件的成本下界，不是所有树宽/硬件下训练必定无效的证明。

如果有合适的条件分布，KL 可约束局部差异；但闭环状态会因动作变化而改变，接触具有敏感性，长尾失败还改变运行时长。因此小 KL 不能自动保证环境 trace 长度接近。当前 deterministic direction 并没有可直接套用的离散 softmax 概率；若人为定义 Gaussian surrogate，需说明噪声尺度、密度支持与校准，并继续用真实任务指标验收。

## 5. 最值得借鉴的训练改进：显式世界条件，而不是先动 target

### 5.1 本地代码依据

[当前训练入口](../../../review/OMISSIONS.md) 的 `compact`、`stack`、`decode`（21–29、118–130 行），以及 [部署 predictor](../../../投机采样/scripts/p0_batched_verify.py) 的 `predict`（303–354 行）显示，显式输入为：

\[q_\phi(x_j, d_{j-1}, \sigma_j,\sigma_{j+1}, t_j).\]

`state` 是动作去噪 latent 的紧凑表示，不是完整环境状态；没有显式图像、视频/世界 latent 或 target hidden 输入。**上一 target 方向会间接携带条件信息**，所以不能断言“完全看不见场景”；但 WAM 自身有世界条件，不代表小 draft 已能充分访问这些信息。

### 5.2 最小可研究假设（未授权实现）

优先候选是：

\[q_\phi(x_j,d_{j-1},\sigma_j,\sigma_{j+1},t_j, z_{world},h_{target}).\]

这里只表示可选条件，**不是现在承诺把所有特征都接入**。先 CPU 审计：哪些特征已在固定 target 原计算路径里存在、可因果获取、可稳定记录、能按 chunk 复用；复制/池化/offload 的额外成本也必须计入。若需要先生成额外未来视频，便不再是“免费复用”。

建议后续训练分离三个问题：

1. **对照**：原条件 draft，冻结 T、阈值、数据/更新预算及评估口径。
2. **条件改进**：增加已有世界/target 条件；容量、额外输入及读取成本单列，尽量用同结构零条件/置乱条件消融排除单纯参数增多。
3. **辅助损失**：在条件改进有收益后再加未来 latent/接触阶段等监督；未来信息仅用于训练标签，不能泄露到推理或评估划分。

数据应包含当前候选真实访问的状态、拒绝节点、困难抓取/恢复阶段；训练/验证按 episode/init 划分，不按相邻去噪行随机切分。采集失败和 fallback 也收费，不能只留成功 easy states。

这是 DistillSpec、在线蒸馏和已有特征 drafting 的组合启发，而非“照抄 WA”。此前 B_on 是固定 S0 访问状态的一次更新，缺少反复候选自身分布适配；其失败足以阻止盲目扩训，不足以否定所有 on-policy 蒸馏。[DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)、[Online SD](https://proceedings.mlr.press/v235/liu24y.html)、[EAGLE](https://proceedings.mlr.press/v235/li24bt.html)

“真实 LIBERO 训练还是模仿大模型”不是二选一：**真实环境提供被访问的状态和行为结果，冻结 target 提供这些状态上的局部监督；必要时再加任务/阶段约束**。改变参考策略质量与加速同一参考策略是两个独立实验。

## 6. RL、KL 和大小模型协同：值得研究，但顺序靠后

- **先做成本感知启停/预算选择**：KV/history、当前需求、验证成本、预计提交收益等决定用不用投机或用多大树；先简单可解释规则，不先训练 RL 调度器。Learning To Draft 证明这个研究方向已有正式文献，不证明现成调度器适用于 WAM。[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)
- **再考虑任务 RL 的必要性**：要同时有真实成功信号、失败成本、可靠密度或其他合适训练估计，且算上 rollout/GPU 代价；只奖励接受长度会诱导昂贵候选或偏离动作。机器人 DPPO 是参考，不是当前 flow 小模型的即插即用算法。[ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html)
- **KL 是辅助项，不是环境步数护栏**：实际 gate 仍是成功率、配对环境步数/尾部、抓取返工和完整耗时；可能加入任务条件的 penalty，但不能把形式上一个 KL 当成 trace 保证。
- **冻结 target 仍是当前首选**：大小模型协同可以指共享条件、在线 teacher 反馈、成本调度，不一定指同时更新两套模型。若真更新 target 权重，需新设 T_new 的未投机基线，旧 T 原基线继续保留；不能把 target 变强贡献算入 draft 加速。

## 7. 要不要重构：判据比直觉更重要

| 选项 | 当前建议 | 理由 |
|---|---|---|
| 原样继续，只优化 L2/接受率 | 不推荐 | 已证实调用减少但候选行负担增加；行为也不是 L2 单调函数 |
| 保留架构，成本优先、补场景条件 | **推荐** | 与现有可复现/冻结对照兼容，实验可小步归因 |
| 完整复刻 WA，新 encoder+LoRA target | 当前不推荐 | 替换 target 和问题定义，尚未排除现有计算不回本 |
| 立刻 RL 或大小模型联训 | 当前不推荐 | 成本/数据/概率定义尚未成熟，难以解释收益来源 |
| 转异步执行/长 chunk/strict stochastic branch | 条件触发才讨论 | 这些会改变执行、参考采样器或协议，不是现有成本关的一项无扰动修改 |

真正考虑架构级重构的触发条件：固定负载显示长历史下合理候选预算都不能回本、廉价场景条件不可获得，或跨任务质量退化无法用局部训练/保守启停控制。当前四格成本尚未做，不能提前宣布触发。

我们的潜在贡献不是“首次世界模型+投机”，也不是“树更复杂所以更好”；邻近工作已相当密集。可能有价值的研究问题是：**连续 flow 的世界条件投机如何同时做到历史/并发成本可回本、接触阶段行为可靠和真实闭环 E2E 改善**。这是待验证选题，不是已有创新或 CCF 投稿保证。

目前主要证据仍是一个任务/少量初态。发表级结果需要独立多任务、多初态、失败计费、强基线、置信区间、非劣口径以及完整任务耗时；两个匹配成功回合不能代替这些。

## 8. 下一步：先保持主线，预算仍未批准

1. **本轮完成**：保存进展检查点、文献调研、发表状态区分、方案建议及 HW ledger；不启动任何新 GPU/EGL/服务/episode/teacher/训练，不改代码/权重/阈值/公共模型/保护状态或 tile。
2. **待用户确认**：是否采用“不重构、成本关优先，世界条件训练后置”的研究建议；这不自动批准 GPU 或新训练。
3. **若继续成本关，仍需明确确认原候选预算及实现范围**：0 新完整 episode、最多 2 次双 EGL 初态初始化、168 固定 chunk 上限；GPU 预计 35–55 分钟、90 分钟硬安全限时；仅任务专用默认关闭 snapshot/replay/监督器/CPU 比较器，先 CPU 工程预检，再资源/fixture 精确恢复及成本测量。总流程预计 90–165 分钟，非保证。
4. **关卡通过以后**：先根据真实成本决定是否缩树/启停等下一最小实验；场景条件训练须另给方案、数据预算与成功/步数/E2E 晋级标准，不自动开训。

详细通过与停止标准保留在 [既有固定负载方案](FIXED-LOAD-COST-PLAN-20261011.md)。失败停、不补局、不提高容忍、不自动改自然 E2E 或旧 8 局入口。

## 9. 调研证据与写入边界

- [书目、发表核验层级](../../../review/OMISSIONS.md)
- [关键说法核对、推论及非结论](../../../review/OMISSIONS.md)
- [写入前旧文件保护清单](../../../review/OMISSIONS.md)
- [HW 增量写入日志](../../../review/OMISSIONS.md)
- [写后 CPU 保留/一致性检查](../../../review/OMISSIONS.md)

书目为人工基于原文/正式论文集的核验记录，不是所有网页的完整归档，不构成第三方复现实验。CCF 站点存在访问验证，目录正文使用高校保存的 CCF 正式文件核对；发表状态以可验证原始证据分层，不从没有搜到接收反推拒稿。

并发保留说明：首次写入保护检查发现另一会话更新了 3 个 tile 文档，拦截发生在共享写入之前。本任务未写这些文件、未回滚其更新；初始清单其余 1620 项不变，随后建立最新写入前清单，并对共享 HW 做增量追加/字节重建检查。[并发变更记录](../../../review/OMISSIONS.md)、[最新写入前清单](../../../review/OMISSIONS.md)。这只是写入安全记录，不将 tile 进展纳入本研究。
