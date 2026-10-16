# 投机采样：本机正式计时、AutoDL 质量测试、空间清理与下一步

> 15:10追加：更完整的E2E端点、reset分解、全部正负优化与采集提速结论见[新统一报告](PROJECT-STATUS-E2E-AND-THROUGHPUT-20261015.md)。本文14:46资源进度保留为当时快照。

更新时间：2026-10-15 14:46（Asia/Shanghai）。本文是本轮状态快照；运行中任务的终态以各 job 的 `FINAL-ANALYSIS.json`、`RUN-*-RESULT.json` 和 `status.json` 为准。tile Transformer 算子源码及其 HW 记录未改动。

HW Knowledge Ledger：`CHAT-20261015-LOCAL-AUTODL-CLEANUP-PLAN` 与 `CHAT-20261015-OFFLINE-SOURCE-AND-A-CONF-LITERATURE`；六类索引与 compact 已同步。

## 已固定的方法与证据边界

当前运行候选仍是 A_off MLP draft、K2/B2 bonus、`prefix_then_cost`、阈值 `0.007839680526450805`、共享 KV 与 O1–O3。当前边成本是同查询一步 Euler 误差 `|Δσ|·||d_draft−d_target||₂`；阈值来自旧 task8 两层边误差合并分布的 q90，**不是任务成功率 90% 的保证**，详见[原始溯源](DRAFT-DESIGN-ACCEPTANCE-LITERATURE-20261013.md)。TF64（单层、宽度 64、4 头、53,070 参数）和匹配 MLP96 的旧三 seed 离线训练均未过原准入门；没有选定新 Transformer 权重，也没有宣称 Long10 成功率非劣。短/中/长 KV 的完整 policy chunk 已有 1.602×/1.544×/1.500×，18/18 正式配对更快；此结果不含 reset、环境和整任务。

现有 [TF64 源码](../../../投机采样/experiments/draft_architecture_v1/models.py)把 112 维动作状态和上一方向按 7 通道×16 位置重排，每位置附加 `σ、next_σ、timestep/1000` 三标量形成 17 维输入；经 64 维嵌入、1 层 4 头自注意力/256 维 FFN、双方向头输出两条 112 维候选。它**没有文本 cross-attention**。若数据门通过，下一结构对照可复用冻结 target 的 T5 指令 embedding，缓存/池化为少量提示词 token，让 16 个动作位置对它们做单层 cross-attention；同输入查询、参数量与训练预算匹配 MLP 对照，并计入 reset 编码和每 chunk cross-attention 开销。这是待验证结构，不是目前已训练或已部署的模型。

已有接受规则对照给出排除依据：[夜间完整结果](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)中，同“最长合法前缀再最小成本”目标的 DP 与现有穷举在 2,920 棵树上完全相同，换 DP 本身无算法收益；greedy 在 4 个开发配对里虽把 target 调用从 OLD 的 3,527 降至 3,492，但 policy chunk 也从 72 变 71，单 chunk 调用反而略高，不能声称 verifier 更高效。严格零误差接受门在 3,906 个 TRAIN 查询上 0 行通过，失去投机收益。MSE/RES 阈值在四条件闭环中没有同时通过 T 与旧 WL2 的预设质量门。PF2/3/4 同阈值周期刷新在两个复用开发配对都成功，但比连续投机 OLD 多 13.38%/7.52%/4.64% target 调用，尚无优于 OLD 的证据。故现阶段冻结 OLD/A_off；新门控须先在 CAL 与跨任务闭环证明质量和总耗时，不能只凭接受率或这几个小样本换配置。

本机 [单 task 正式 E2E 审计](../../../投机采样/results/runtime/p0-formal-e2e-speed-v1-single-r2-20261015/FINAL-ANALYSIS.json) 完成 task0/init25–28 的 T/A_off 四对、8 条完整轨迹，外来 GPU 进程为零。T 成功 3/4、A_off 成功 4/4；双方都成功的前三对自然 episode 汇总比为 `760.179/639.418 = 1.188861×`，冷启动到 client 退出比为 `880.088/757.132 = 1.162398×`。前三对 T/A_off 的环境动作步数为 818/834，policy 请求时间为 345.892/218.149 秒，reset 请求时间为 338.702/344.500 秒。第 4 对 T 达 796 步未成功、A_off 293 步成功；把它纳入全部 episode 可得 1.4669×，但该值混入了 T 失败的长时间，**不能解释为投机的性能收益**。四对仅单 task，不能证明总体成功率。

本机多 task 正式计划为 task0/1/5/9 各 init29、T/A_off 交错共 8 条。第一 job `/var/tmp/cxz-formal-e2e-speed/p0-formal-e2e-speed-v1-multi-20261015` 完成 3 条后，在 task1 的 T 轨迹内检测到外来 `lhw` 进程 PID 2138101、显存 450 MiB，按独占合同中止并保留 `FAILED.json`；自有模型、EGL 与 runner 已退出。随后查到 `lhw` 的另一进程 PID 2215798，命令为 `work/perf-investigation/refresh_contact_build.py`，约占 3.8 GiB。新 job `/var/tmp/cxz-formal-e2e-speed/p0-formal-e2e-speed-v1-multi-r2-20261015` CPU 准备后，在短暂空闲的自动准入时又遇到 `lhw` 新 PID 2259873，正式入口于第 1 条前中止、0 新完整 episode；两次失败都保留且自有进程清理完成。用户随后授权“GPU 空了就跑”，故 14:02 又用**新** job `/var/tmp/cxz-formal-e2e-speed/p0-formal-e2e-speed-v1-multi-r3-20261015` 从头执行。该 job 已完成全部 8 条，独立 CPU [正式配对审计](../../../review/OMISSIONS.md)返回 `PASS_LOCAL_FORMAL_E2E_PAIRED_AUDIT`：496 条 GPU 遥测无外来进程，四对初态哈希一致，目标调用计数闭合、无 horizon 中途截断、自有 GPU 进程清理后 `nvidia-smi` 无 compute PID。前两次失败与这次通过均保留，**没有拼接旧样本**。

四对 T 与 A_off 均成功，双方总环境步数分别 1,054/1,044，总 policy chunk 分别 69/68。双方都成功的自然 episode 汇总耗时 `1033.403/898.071 = 1.150692×`；冷 server 启动到 client 退出 `1192.672/1093.244 = 1.090947×`。四个任务的自然 episode 单对比分别为 task0 `1.2462×`、task1 `1.1818×`、task5 `1.0703×`、task9 `1.1065×`；环境步数 T/A_off 分别为 `287/272`、`264/262`、`194/185`、`309/325`。policy 请求合计 `456.802/298.572 = 1.5299×`，reset 请求却是 `483.740/505.138` 秒；当前投机主要节省 policy，未缩短 reset。四对自举 95% 比值区间 `[1.0899,1.2142]` 只表示此极小样本的计时波动，**每任务仅一个初态，不能用来保证 Long10 成功率或轨迹长度**。即便四对全部成功，task9 的 A_off 比 T 多 16 环境步，说明不能只看总步数。正式多任务速度可作为初步主表，质量推广仍以 AutoDL 更大配对为准。

## AutoDL：正式质量测试

旧 `/root/autodl-tmp/libero-scale/train194-job` 原计划 194 条完整 A_off on-policy TRAIN；完成 10 条后，task0/init13 的 `nvidia-smi` 15 秒超时使本次 run 标为 FAILED，旧 `RUN-0001-RESULT.json` 与不完整 episode 均保留，所有自有进程已回收。不能把 10 条当作完成的 194 条训练集。源码 `libero_scale_v1/run.py` 的 GPU 采样已改为连续三次超时才判失败，新源码已同步远端；旧运行证据未覆盖。

在用户约一天 AutoDL 时限和 19 GiB 可用数据盘约束下，启动了新的 **质量优先** 计划 `/root/autodl-tmp/libero-scale/heldout50pair-job`，计划 SHA-256 `5a5b11feeef064fb068390187ad7f3054fffac4a5a605077084b7c2ae6f4c924`。Long10 task0–9、init25–29，T/A_off 共 50 对、100 条完整 horizon，严格配对，CPU 正式 preflight 为 PASS；北京时间 13:35 启动远端 PID 13192。14:37 进度 9/100 条，尚在 task0，不能拿前几条推断 Long10 效果。按当前吞吐，完整结果预计 10 月 16 日北京时间凌晨 02:00–06:00；跨任务/horizon 波动与异常可能延长，以 60,000 秒 GPU 硬上限为准。输出上限 8 GiB、磁盘最低 8 GiB；远端 A100-PCIe-40GB 的时间**不用于论文速度表**。此组是协议内留出描述性质量筛查，历史 target 元数据已暴露，不称严格盲测。没有运行新的 TF 训练，也没有启动备用 train194-r2。

## 本机空间清理与保留

清理前项目约 184 GiB、`/rjs` 可用不足 0.1 GiB；本轮后项目约 149 GiB、`/rjs` 可用约 42 GiB。实际执行了三个边界明确的清理：

逐路径、字节数和清理后缺席核对见[执行清单](../../../review/OMISSIONS.md)。

1. 用 `pip cache purge` 删除用户 pip 下载缓存 1,176 项，工具报告 6,992.7 MB；不涉及环境包。
2. 删除旧 O1/O2/O3 四个 job 中精确清单上的 14 个 `*-complete-chunk.pt` 历史张量快照，逻辑字节 22,971,099,746（21.39 GiB）。删除前按候选清单验证文件名、所属 job、大小、普通文件和单链接；保存原计划、运行结果、失败账本、源码和 HW。用户已明确允许重采样算子数据；若要重做这些旧 job 的逐张量比较，需重新采样。
3. 删除 2026-09-22 旧 Hugging Face 下载目录中的 3 个 `.incomplete` 文件，逻辑字节 14,740,903,548（13.73 GiB）；它们不是已加载的模型权重，当前任务使用完整 ModelScope Long 检查点。完整权重和目录内 metadata 保留。

读盘后其余大头分为：模型约 53 GiB（Long 完整权重和 RoboTwin 权重，保留）；两批 target-only `server_output` 约 17.7 GiB 及对应逐 chunk trace 约 3.7 GiB（离线教师来源，保留）；旧 formal1 trace 约 10 GiB（训练数据溯源，保留）；phase4 BF16/FP32 视频 draft mmaps 约 17.05 GiB（当前 action 论文不调用，但若继续视频分支需重新构建，暂保留）；`p0-fixed-load-cost` 真实 fixture 与后态约 11 GiB（旧固定负载源码与原性能核验仍引用，暂保留）。不因“旧”字样删除具有来源或复现用途的数据。tile 方向内容、受保护 HW state/cycle/rules、模型、历史报告均保留。旧[清理提案](CLEANUP-PROPOSAL-20261014.md)是 10 月 14 日时点的提案，以上是 10 月 15 日用户授权后的实际执行记录。

## 文献依据与方法取舍

本轮另对[第一批 target-only 记录](../../../review/OMISSIONS.md)与[连续运行记录](../../../review/OMISSIONS.md)做了只读去重：前者 266 条，后者 493 条且**包含前者全部 266 条**，所以总共是 493 个唯一 `(task,init)`，不是 759 个。缺的是 task8/init43–49；task8/init42 有一条 `status=error`。[元数据来源清单](../../../review/OMISSIONS.md)将其余 **492 条完整记录逐一唯一对应**到提示词、client 起止时间窗内的服务端目录，检查每条 `actions/latents` 的完整 4 帧递增序列；467 条的观测比 chunk 少 1 个，另 25 条保存最后一帧观测，没有缺帧或目录复用。还将完整 episode 按 client 顺序与三份逐 chunk trace 匹配：第一批 4,734 份、延续 worker0 2,005 份、worker1 2,649 份，共 9,388 份均按 episode 内 `frame=0,4,...` 连续；worker0 另有末尾 6 份无法归入完整 episode，邻接 task8/init42 的失败记录，**不计入训练**。三个 trace run metadata 均写有 `wam-p0-trace/v3`、`draft_train_v2_transformer_output`；client metadata 标明 Long 官方 checkpoint 的加载路径，但本次未复核旧运行时权重字节。按当前协议切分，TRAIN 194/194 完整、3,672 policy chunk；CAL init20–24 50/50、1,005 chunk；本次质量组 init25–29 50/50、966 chunk 均有历史 T 记录。这个审计**仅证明元数据、路径和帧序列**，没有读取 tensor 或重建当前 A_off 的 same-query verifier 标签。它是可追踪的 target-only 离线预训练候选；后续须对 TRAIN trace 的状态/方向/σ 和 checkpoint 做数值核验，若要当前 A_off 同查询标签仍须在候选访问状态调用 teacher；CAL/质量目录不得用于训练。

- [KERV，DAC 2026](https://arxiv.org/pdf/2603.01581)及其[训练说明](https://github.com/zhengzihaoPKU/KERV/blob/master/docs/TRAINING.md)：用冻结 OpenVLA 在 LIBERO 示范数据的每一步离线生成 hidden state/action-token 教师样本，再训练单层 draft；每任务 50 次仿真闭环测试。它根据运动学变化调整接受阈值；在 LIBERO-Long，把静态阈值从 9 放到 15/20 时，报告的成功率与推理步数明显恶化。这是**离线训练可以替代昂贵的仿真采集**以及**接受规则须看控制风险**的依据，但其离散 token 阈值不能直接照搬到 WAM 的连续 flow direction。
- [Spec-VLA，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1367.pdf)：从 LIBERO 示范经冻结 OpenVLA 重新生成 draft 训练数据，再在每任务 50 次 rollout 中测成功率；放宽接受阈值提高接受长度，但 Long 套件使用比其他 suite 更小的阈值。其表中 `Length` 是单次前向预测的动作 token 数，**不是环境轨迹步数**。
- [Group Tree Optimization，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6e67ae290635d0874c4cb43ba2a2cfb-Abstract-Conference.html)将训练目标对齐实际 tree decoding 的期望接受长度；[Learning to Draft，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)直接优化 draft+verify 周期吞吐，而非只优化接受率；[DFlash，ICML 2026](https://proceedings.mlr.press/v306/chen26bv.html)用 target 上下文特征条件化轻量 draft。三者支持下一轮做 tree-policy、真实成本和条件输入对照；它们的 LLM 保真结论不自动证明机器人闭环成功。
- [FlowCast，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)针对连续 flow 使用速度外推、MSE 接受阈值，并在 Lipschitz 等假设下界定生成轨迹偏差。这给 WAM 的 σ 加权方向误差和累积状态漂移实验提供直接理论参照，但论文实验是视觉生成，其界**不是**环境轨迹长度或机器人任务成功率保证。可复现方向是把其零参数外推作为同查询离线对照，而不是照搬阈值。
- [SP-VLA，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4072543747a14bbed76284cf2c04b9e9-Abstract-Conference.html)按动作性质在轻量生成器与完整 VLA 之间切换，报告 LIBERO 约 1.5× 加速且维持其测试中的成功率；[官方实现](https://github.com/ChildTang/SP-VLA)公开了基于历史动作、竖直位移等条件的 step skipping。它支持把抓取/放置等关键阶段的 target 回退纳入本项目接受器对照，但 OpenVLA 的场景与 WAM 连续 flow 不同，应在同一 frozen CAL 上重新定门并统计 guard 开销。
- [BAC，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2712b17bb58ea5b2b65c45857b024744-Abstract-Conference.html)对机器人 diffusion policy 做 block 级自适应缓存，强调跨 block 误差传播会放大；这是 MegaTransformer/verifier 数值审计和分层调度的相关对照，不等于本项目已有算子收益。
- [LingBot-VA 官方仓库](https://github.com/robbyant/lingbot-va)提供 LIBERO-Long 与 RoboTwin-2.0 的 posttrain 权重/数据/评估入口，未列 RoboDojo 权重。第二仿真优先 RoboTwin；先完成本模型的 Long10 质量链，再扩 benchmark。
- [LingBot-VA 原论文，RSS 2026](https://www.roboticsproceedings.org/rss22/p016.pdf)清楚区分训练与测试：先做视频/机器人数据预训练，LIBERO 四套件各用每任务 50 条示范微调；每套件最后做 **3 个 seed × 500 次闭环评估**。原模型用冻结 T5 文本编码，并在 Transformer 里以 cross-attention 融入指令。这说明“在训练集训好”不能代替 LIBERO 成功率，且为下一版 prompt 条件化小 TF 提供现成文本特征入口；我们加速的是冻结的 posttrain 目标模型，不能把重训原 WAM 与训练 draft 混为同一步。

论文定位必须避开已有工作：KERV 已有运动学动态阈值，SP-VLA 已有关键动作阶段的轻重模型切换，FlowCast 已有连续 flow 的投机阈值与生成偏差分析。因此“用小模型预测、设阈值接收”和“某些动作调用大模型”本身不足以作为 CVPR 新贡献。可成立的独特主张须由实验证实：在 WAM 的连续动作去噪/共享 KV/B14 批量 verifier 上，提出可审计的候选路径与风险门控，给出同查询训练和闭环成功／环境轨迹／自然 E2E 的完整证据；若 MegaTransformer 进入主贡献，还需独立的同输出数值与净 E2E 收益。没有这些表时，应缩小题目而不是靠单 chunk 加速或训练误差代替。

## 倒排计划与判定点

| 时间（北京时间） | 必交证据与并行工作 | 决策条件 |
| --- | --- | --- |
| 10/15–16 | 本机独占窗口完成多 task 正式 E2E；AutoDL 完成 50 对质量筛查。输出逐 task 成功/失败配对、双方成功的环境步数比 P50/P90、全样本 horizon 惩罚步数、自然 episode、accept-depth 与 target intents。 | 50 对只筛严重退化；不能据此宣布 3 个百分点成功率非劣。任何失败均保留在分母。 |
| 10/16–18 | 优先审计现有 194 条 TRAIN、3,672 个 chunk 的 target-only observation/latent/action 与原 `(task,init,chunk)` 的唯一映射，再按同一 target checkpoint、prompt、KV、噪声、σ/timestep 离线重建 verifier 教师查询；若复原失败，转用官方 LIBERO 示范数据。冻结 FIT/CAL 划分与无泄漏清单；实现只读离线教师生成和 episode 级数据索引。 | 同查询标签、来源、输入范围和在线接口通过审核，才投入完整 draft 训练。纯 action BC 无法监督当前 verifier 查询；若采用全部旧 493 条训练，将泄漏质量/未来留出，必须严格只取 TRAIN。 |
| 10/18–21 | 在同一数据/种子/预算下训练 A_off 参照、容量匹配 MLP、TF64 和轻量 prompt 条件化 TF；衡量 root 保真、tree 接受、实际 draft+verify 周期成本，CAL 上选一次模型和接受规则。 | TF 若继续未过离线和独立闭环门，论文基线仍用 A_off；不要为了架构叙事替换更可靠的方法。 |
| 10/20–26 | 对静态阈值、σ 归一误差、动作语义守卫、运动学动态阈值和风险阶段回退做**预先固定**的有限消融；在 DEV/CAL 决策后冻结测试配置。同步对当前 verifier 做真实 profile、单 block 数值与成本优化，隔离 MegaTransformer；原 tile 路线不动。 | 成功率与轨迹长度优先于 acceptance；新规则要带上自身开销，最终比较整任务 wall time。 |
| 10/24–31 | 冻结后跑 Long10 扩大配对质量，至少把 init30–49 留作本轮未用的协议留出；标准 10×50 表可作描述性 benchmark，并附 task 分层配对区间。条件允许再用官方 RoboTwin 权重做第二模拟器；补单 chunk、单 task、自然 E2E 和 verifier 消融图。 | 是否有充足独立样本由 50 对中的不一致成功率估计再定；不能将同一批筛选数据重新包装为 blind TEST。RoboDojo、真机与 PF3/4 不排在主证据链之前。 |
| 11/01 18:00 | 将完整中文稿和现有主表交老师；方法、系统、实验协议现在即可逐段在会话中写，论文稿不落本地文件。 | 若 10/22 前还没有跨任务闭环质量，缩小论文主张；MegaTransformer 可作为独立系统扩展，不以未证实收益充主结果。 |

中文稿可从现在按论文实际章节逐段讨论，用户自行放飞书，本地只保留研究计划和结果表：①引言明确连续动作 flow 推理代价、任务质量约束及贡献边界；②相关工作分连续生成投机、VLA 动作调度、diffusion policy 加速；③方法写 WAM 状态/σ/timestep、双候选 draft、K2/B2 树和最长合法前缀、接受误差与 fallback 的严格伪代码；④系统实现写共享 KV、B14 verifier、目标调用成本，并仅在数值/E2E 验收后加入 MegaTransformer；⑤实验写 LIBERO 成功率/环境步数/速度四层表、阈值/结构/树/PF/guard 消融、RoboTwin 跨环境和错误案例；⑥局限列出近似接受不能保证逐步同轨、历史初态暴露、单卡与仿真边界。10/18 前可先完成③的中文逻辑草稿和⑤的空表定义，10/25 前用数据改写引言与摘要。

执行时优先使用三个事先固定的判定门。第一，数据门：只读回放 TRAIN 的原观测、动作与 latent 后，必须核验 target checkpoint、初始观测、每个 chunk 的当前状态与 σ/timestep、旧源和新查询的标识；只要同查询对不上，就把旧数据限定为预训练，不标成当前 verifier 标签。第二，模型门：对 A_off、容量匹配 MLP 和 TF 使用相同新 TRAIN、归一化、三个种子、真实 draft+verify 成本，并在 CAL 检查 root/分任务失守率；TF 必须先过冻结准入再上闭环。第三，质量门：公开每 task 的成败配对四格表及失败/800 步截断，计算配对成功率差的 task 分层置信区间、双方成功的环境步数比 P50/P90、全样本 horizon 惩罚步数、自然 E2E；仅在正式评估前固定阈值和规则。50 对是发现明显退化的筛查，不能据此声称 3 个百分点非劣。

样本量也有硬边界：即使 50 对里一次“原 T 成功、A_off 失败”都没有，按独立配对的一侧 95% 二项上界，损失概率仍可达 `1−0.05^(1/50)≈5.82%`；在相同零损失的理想情形约需 **99 对**才把这个上界压到 3% 以下，出现损失则要更多。任务内仅 5 对的本轮 AutoDL 结果尤其不能当逐任务保证。因此 50 对通过后，应在冻结算法后扩大到预先登记的 250 对或按实际不一致对数重算样本量，失败与超时不可删；最终统计还应给出 task 分层与成功率净差，而非只看单向损失。

“trace 长度”在本文实验表中明确拆成三种量：单 chunk 内固定的 50 次 flow 更新、每任务实际 policy chunk/target 调用数、以及机器人执行至成功或 800 步 horizon 的环境动作步数。FlowCast 的理论界只触及第一种生成状态偏差；KERV/Spec-VLA 部分论文的 decoding/inference `Length` 是第二种；用户关心的任务轨迹必须以第三种并结合成功率衡量。没有论文能让当前连续控制系统只凭训练误差保证第三种量逐条不变，故不设“轨迹严格相等”的虚假验收门。

接受器只安排一个可解释的有限对照：当前固定 `prefix_then_cost` 和同查询 `|Δσ|·||d_draft−d_target||₂` 门为参照；FlowCast 式**零参数速度外推 draft**及累计误差预算（本项目现有单步 Euler 度量与其误差形式相关，不算一个新发明）；KERV 式由动作空间运动变化调节阈值；SP-VLA 式在抓取/放置、接触变化等关键动作段强制 target；以及前述规则的有限组合。每种都先核验相同 target 查询、候选树路径和回退动作，再在 CAL 固定阈值，最终同时报告接受深度、target 次数、guard 自身成本、环境步数和成功率。PF3/4 仅在成本模型预测收益且 CAL 没有质量退化时进入较贵的闭环；否则保留 PF2/当前 K2/B2。KERV 与 SP-VLA 有[官方代码](https://github.com/zhengzihaoPKU/KERV)、[官方代码](https://github.com/ChildTang/SP-VLA)，可复现其规则思路；其原权重、离散 token 场景和 WAM 参数不同，不把对方发表的数值当本项目复现实验结果。

并行边界：AutoDL 当前只运行一套 50 对质量 job；在磁盘余量、单卡吞吐和剩余租期明朗前不叠加 194 条采集或新 TF 训练。本机空闲窗口可跑单独多 task 速度；该结果不与 AutoDL 的计时混表。CPU 端可同步做来源回放审核、论文中文方法段落和接受器 CAL 分析。MegaTransformer 只对冻结基线 verifier 做隔离 profile/单变量原型与数值检查，等投机质量链明确后决定是否作为本文贡献；原 tile Transformer 路线和 HW 保持不动。

AutoDL 的真实资源账需要单列：前 4 条质量轨迹约 26 分钟，100 条按此量级约 11 小时，计划硬上限 16.7 小时；旧 TRAIN194 在 10 条处失败，余下 184 条同类完整 on-policy 采集即使按每条 6 分钟也约 18 小时，**不能塞进这次一天租期的剩余窗口**。因此本轮先交质量筛查；TF 的扩大训练先用已索引的历史 TRAIN target-only trace 做明确标注的预训练研究，缺少的候选状态同查询数据再按准入必要性分批采，若需要全 194 条须另安排租期或更高吞吐的正式采集方案。这个估算随任务/horizon 变化，不能当完成承诺。

[CVPR 2027 官方时间](https://cvpr.thecvf.com/Conferences/2027/Dates)：论文注册 11/10 AOE（北京 11/11 19:59）、正文 11/16 AOE（北京 11/17 19:59）、补充材料 11/23 AOE（北京 11/24 19:59）。官网未列独立 abstract 日期；项目内部把注册日作为标题/摘要/作者定稿硬线，并在 11/08 前完成登记检查。
