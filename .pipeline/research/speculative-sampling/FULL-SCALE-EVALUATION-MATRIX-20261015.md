# 投机侧冻结后的全量评测矩阵（2026-10-15，北京时间）

> **17:44追加：** 本机reset八完整任务、历史179928行数值审核与TF64/MLP96六公平预训练已完成，新的四臂40完整DEV运行中，详见[本轮结果](LOCAL-234-RESULT-20261015.md)。下文“TF尚未过离线六门/数据待采194/自然E2E待测”等保留为早间快照；当前TF全三seed通过旧六门诊断，但完整闭环与扩大质量仍待完成。下一版应使用已验收bank和常驻模型的独立协议，不能直接将旧fullscale训练/配置入口视为兼容新target-trace checkpoint。主表口径、配对统计与不以离线误差替代成功率的要求仍适用。

## 证据边界与当前决定

当前冻结的全量采集候选是 `投机采样/configs/current_speculative_candidate.json` 中的 A_off、K2/B2 bonus、`prefix_then_cost`、阈值 `0.007839680526450805` 与共享 KV/O1–O3。TF64 尚未通过离线六门；r9 在用户要求停止小规模试跑后只留下 1 条完整 T 历史局，不能据此宣称成功率非劣或升级权重。今日冻结的是用于**数据采集和下一轮比较**的算法、权重 SHA、目标 checkpoint、接受规则与代码 SHA；正式成功率结论仍必须等全量完整 episode。

Spec-VLA（EMNLP 2025）把 LIBERO 四 suite 的成功率与每前向生成的动作 token 数、生成加速一起报告，每 task 50 次；该文的 Length 不是环境完成步数。其 relaxed 接受下 Goal 78.0→74.4%、Object 89.0→85.0%，说明总体“质量不降”的摘要不能代替分 suite 结果。[论文 PDF，表 1、表 2 和图 4](https://aclanthology.org/2025.emnlp-main.1367.pdf)。FlowCast（ICLR 2026）给出基于速度 MSE 的接受与阈值质量/速度曲线，但目标是图像、视频生成，不能直接证明机器人闭环质量。[会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)。

LingBot-VA 原论文在四个 LIBERO suite 上分别用每 task 50 条示范训练，再以每 suite 每 seed 500 次闭环 trial、三个 seed 报告成功率；训练集拟合并没有替代闭环成功率。[RSS 2026 正式论文，第 6 页](https://www.roboticsproceedings.org/rss22/p016.pdf)。官方仓库公开了 LIBERO-Long 与 RoboTwin 后训练权重及数据入口，没有列出 RoboDojo 后训练权重。[官方权重清单](https://github.com/robbyant/lingbot-va)。

新增直接相关的 CCF A 论文 [KERV（DAC 2026）](https://arxiv.org/pdf/2603.01581)；[CCF 第七版目录](https://www.ccf.org.cn/Academic_Evaluation/ARCH_DCP_SS/)将 DAC 列为 A 类。KERV 用单个 LLaMA block 作 OpenVLA token draft，在每 task 50 次 LIBERO 闭环中同时报告成功率、加速、接受长度与平均推理步数；阈值根据机器人运动学变化调整，参数先按环境采样建表。其 LIBERO-Long 表中固定宽松阈值 `r=9/15/20` 的成功率分别为 49.2/36.2/25.8%、论文所称平均推理步数 408.9/439.6/464.6；动态方案为 48.8%、391.2 步。该表证明单看接受长度会掩盖任务质量与步数增长，也提供“校准+运行时自适应”的论文先例；其 Steps 不能直接与本项目的 LIBERO `env_steps` 数值比较。KERV 的 Long 成功率相对其 naive SD 基线 50.2% 下降 1.4 个百分点，相对无 SD 的 AR 基线 54.4% 下降 5.6 个百分点，不能只引用“几乎不损失成功率”的摘要。它是 OpenVLA 的离散动作 token，不提供连续 WAM Euler 误差阈值的可迁移数值。论文明确没有做真机适配，所以仿真工作也有 A 类先例；CVPR 是否充分仍取决于本项目的视觉/机器人贡献与跨基准证据。

## 论文必需结果

| 表或图 | 配对与测量 | 现有证据／缺口 |
|---|---|---|
| 主质量表 | 同 target 权重、task、init、seed 比较 T、当前 A_off、最终胜出的 draft；各 task 与宏平均成功率、成功↔失败转换数、95% 配对区间；失败/超时在分母 | 只有少量开发闭环；标准 Long10 50 init 已有历史目标运行元数据，当前认证的独立 TEST=0，不能称盲测 |
| 轨迹表 | 环境命令步数、policy chunk 数；双方成功配对的步数比 P50/P90 与区间；全样本按共同 horizon=800 为失败/超时记惩罚步数，另报 horizon 率与最差配对增长 | 现有去噪接受长度不等于环境轨迹长度；不能只筛选双方成功而隐藏失败长尾 |
| 接受机制图 | K2/B2 接受深度 0/1/2、root/第二边按 σ 分层、bonus 命中、回退、目标前向次数、实际 7 状态/14 CFG 行量和显存；校准集画阈值—接受—成功/步数/时延 Pareto | 当前有 chunk 级计数和旧开发数据；缺多任务闭环阈值曲线。阈值只能在 calibration 上选，TEST 不再调参 |
| 速度主表 | 本机独占卡，同一 target/输入、固定短中长 KV 完整 policy chunk；自然单 task 完整 episode 与多 task E2E，分别报 warm/cold、reset、policy、cache、环境与加载，P50/P90/区间/峰值显存 | 当前 A_off 完整 policy chunk 为短 1.602×、中 1.544×、长 1.500×，18/18 对更快；它不含 reset、环境、随后 cache 请求，不能称 E2E |
| 消融 | T；A_off 去 bonus／保留 bonus；固定 draft 改 K/B、阈值、selector、PF 周期，逐次只改一个因子；TF 仅在同数据、容量和闭环门通过后进入主表 | MSE/RES 和 PF2+ 目前不能直接晋级；MegaTransformer verifier 数值及速度需另列算子贡献 |

## 数据划分与放大顺序

1. 保存逐 episode 用途台账：TRAIN 用于梯度、CAL 用于阈值/模型选择、DEV 用于迭代闭环、TEST 只能最后一次读取。按完整 episode/init 分组，跨任务泛化需留出整个 task。task0/1 init8、task0/1 init20、task2–9 init17 不能再写作独立 TEST。历史 500 组合全有完成元数据，仅凭“未进入本次训练”不能称严格盲测；可做预登记的新评估集，但须标明历史暴露。真正盲测需新随机初态或新任务/基准并审核目标模型训练暴露。
2. 在 **A_off verifier 的 on-policy 树**内同查询状态采集 target 方向与 σ、动作状态、上一步方向、prompt 条件和实际选择路径；现有三节点 teacher 钩子不在真正的 T-only 模式产生标签，不能把这种采集称作 target-only 轨迹。按用户要求直接启动完整 Long10 TRAIN 采集，不再新增 smoke/pilot；每条必须完整 horizon、审计全部计数和失败。训练与阈值校准不得混用最终评估初态。TF 文本 cross-attention 需要按相同 tokenizer/prompt 重建并核对冻结 T5 embedding 和 mask；现有 TF64 没有读取 prompt/图像。
3. 开发准入先看闭环成功及新失败，再看双方成功步数。规划用成功率下降不超过 3 个百分点、成功配对步数比中位数不超过 1.10、P90 不超过 1.20 作筛选目标；这些是**待最终 TEST 之前冻结的工作阈值**，当前样本不能证明达标。正式非劣性须预先固定 δ 和样本量，只在 TEST 上用 task 分层的配对置信区间检验：成功率差的单侧 95% 下界大于 −δ。无论门槛如何，保留逐 task 绝对差与长尾。
4. 标准 LIBERO Long10 为 10 task×50 init=500 局/臂。正式评估只跑一次 T/A_off 1000 局 `benchmark500`，从同一批 episode 的 init25–49 切出 250 对协议内留出结果；不再重复跑独立 `heldout250` 的 500 局。两种统计口径共享原始轨迹，不能当作两项独立证据。按历史 266 个 target-only 完整 episode 的均值约 495 秒估算，单臂约 69 GPU worker-hours；单卡两臂约 138 小时、实际需以 AutoDL 运行日志更新。今天只能锁候选、启动和验证流程，不能得到全量成功率结论。新 AutoDL A100-PCIe-40GB 用于数据/训练/算法质量；正式速度只在本机独占 A100-SXM4-40GB 上测，不混用两卡数字。

成功率非劣的样本量需由 CAL 上**配对结果不一致率**估算。若每对的成功差记为 −1/0/+1，不一致率为 `q`，`n` 对时差值标准误近似为 `sqrt((q−d²)/n)`；在 `n=500`、`d≈0` 时，一侧 95% 区间的宽度对 `q=0.10` 约 2.3 个百分点，对 `q=0.20` 约 3.3 个百分点。因此预设 δ=3 个百分点时，500 对未必足以判定非劣；正式计算须按 task 分层并用实际 CAL 不一致率重估，不能在 TEST 失败后修改 δ。环境步数同理报告配对分布和区间，而不把“平均轨迹差不多”当质量保证。

## 今天需要形成的冻结清单

- r7–r9 终态及失败账目完整保留；TF 未达到离线与闭环质量门，冻结 A_off 为大规模采样与评估候选，同时将 prompt-conditioned TF 作为下一轮研究，不因架构偏好替换。
- 写出候选 checkpoint、target **7 份实际加载权重**、接受器、server/client、采样器、数据 split、阈值和环境镜像的 SHA/版本；旧清单中另 2 份位于隐藏下载缓存目录，不属于模型加载清单。远端已完成 CPU 权重/源码校验及 EGL 基础渲染；按用户要求直接启动完整批量任务，不再新增 smoke/pilot。
- 本机等独占窗口测自然任务与 E2E，并把已有短中长完整 chunk 数字仅放在对应列；reset 的高占比在 E2E 分解中如实报告，优化优先级待实测占比和可改性再定。

相关本地证据：[当前完整 chunk 速度](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)、[TF 结构及质量](TF-DRAFT-STRUCTURE-AND-QUALITY-20261015.md)、[代码地图](../../../投机采样/docs/CODE-MAP-20261014.md)。
