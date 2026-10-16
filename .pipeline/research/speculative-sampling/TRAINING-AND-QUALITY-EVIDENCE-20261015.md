# 投机 draft 训练、接受规则与闭环质量：来源及本地证据

核查时间：2026-10-15，北京时间。本文是研究证据记录，不是论文初稿。本轮只读实验源码/产物及公开文献；未执行 GPU、远端写入或新实验。**结论：当前约 95% 的离线 CAL 逐边通过率不足以预测 LIBERO 成功率或环境 trace 长度；先等正在执行的新 30 局严格配对开发实验终态，再决定是否采集新 TF 实际访问候选状态上的教师标签。**

## 五项最相关的正式发表来源

| 来源和正式录用证据 | 直接支持的做法 | 不能转移到本项目的结论 |
|---|---|---|
| [FlowCast，ICLR 2026 正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html) | 无需训练 draft，用锚点速度外推 flow 候选；在候选处查 target 速度，按速度 MSE 阈值接受。论文分析了与原 flow **去噪轨迹**的偏差，并在图像/视频生成实验中报告质量。 | 该界不是机器人环境步数或成功率的界；我们的非均匀 σ、学习型双头 draft、BF16 packed verifier、接触反馈都未被原界覆盖。不能借其阈值声称“保成功率”。 |
| [Spec-VLA，EMNLP 2025 ACL 正式论文页](https://aclanthology.org/2025.emnlp-main.1367/)及[原文表 1、§5](https://aclanthology.org/2025.emnlp-main.1367.pdf) | 用微调后的 verifier OpenVLA **重新生成 draft 训练数据**，而后在四个 LIBERO suite 各 task 跑 50 次闭环；按动作 token 的离散 bin 距离放宽接受，联合报告 SR、每次前向预测 token 数和生成速度。表 1 中 Goal 的原 AR 成功率 78.0%，relaxed 为 74.4%；Object 89.0%→85.0%。 | “Length”是**每次前向生成/接受的 token 数**，不是环境 trace 长度。表内存在成功率下降，摘要的总体“不损成功率”不能被引用成逐 suite 或本项目保证。它是 token 化 OpenVLA，不是连续 flow gate。 |
| [Consistency Policy，RSS 2024 正式 proceedings](https://www.roboticsproceedings.org/rss20/p071.html)及[原文训练/评价](https://www.roboticsproceedings.org/rss20/p071.pdf) | 从预训练 diffusion teacher 的去噪轨迹做 consistency distillation，明确区分单步/多步 student；在仿真任务中以在线 rollout 成功率和标准误评价（Robomimic 最优 checkpoint 在线 200 次），另报 NFE。 | 这是重新蒸馏动作策略，而非保持冻结 target 的投机 verifier；其 teacher 轨迹监督和离线损失也没有推出闭环成功或物理 trace 长度不变。 |
| [SP-VLA，ICLR 2026 正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4072543747a14bbed76284cf2c04b9e9-Abstract-Conference.html) | 按动作阶段在完整 VLA 与轻量生成器间切换，LIBERO/SimplerEnv 均用任务性能、推理频率/延迟核验加速。支持把“只在部分位置用轻量模型”当实证设计问题。 | 它同时改变模型调度和视觉 token 剪枝，不能把其速度或质量数字直接归给我们的 PF2、PF3 或连续 flow 接受阈值。 |
| [DAgger，AISTATS 2011 PMLR 正式论文页](https://proceedings.mlr.press/v15/ross11a.html)及[算法 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | 顺序决策中动作改变后续观察分布；迭代收集**当前策略访问的状态**，在该状态查询 expert，聚合后再训，是处理 distribution shift 的有条件方法论依据。 | 它没有证明我们从旧 A_off 或旧 TF 查询集出发的 500 更新会修复新 TF；其理论假设和 expert-action 闭环形式也不等于我们在 draft branch 上查询 target 速度。这里是采样分布类比，不是直接继承其性能定理。 |

既有 [BAC，ICLR 2026 正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2712b17bb58ea5b2b65c45857b024744-Abstract-Conference.html)是额外边界证据：即使只是 diffusion policy 的 block 缓存，误差也会沿层传播，作者才设计更新调度和误差截断。它不是训练 draft/确定性投机方法；其摘要称“lossless action generation”不能当本项目的轨迹逐位或成功率保证。以上 ICLR 论文的会议身份由各自 `proceedings.iclr.cc` 正式页确认；RSS、EMNLP、AISTATS 分别由官方 proceedings/ACL/PMLR 确认，不把预印本自述当录用。

## 和当前数据的严格对照

- [接受标准审计](ACCEPTANCE-CRITERIA-CCFA-20261014.md)说明当前逐边 `|Δσ|‖v_d−v_t‖₂` 是同一状态、条件下**理想 Euler 一步更新差**，不是最终 packed 输出差，更不是物理环境步数差。现用 `τ=0.007839680526450805` 是历史 task8/init0 两边共 2,400 个误差的 q90，既非独立质量集选出的容差，也非文献给出的安全界。即便零阈值，[先前完整任务审计](TRACE-SUCCESS-CONTROL-RESULT-20261014.md)也见到 packed 与 sequential target 的 BF16/缓存输出差异。
- [原 40 局记录](LOCAL-234-RESULT-20261015.md)给出的 target-only 预训练 TF64/MLP96 离线 CAL 宏任务逐边通过分别为 **95.0425%/95.4111%**。开发集描述性成功为 T 10/10、A_off 8/10、MLP 10/10、TF 9/10；严格同初态的 task3 是 T 264 步成功、MLP 265 步成功、TF 796 步失败。task8 的 T 初始腕相机 hash 与三 draft 不同，整组旧 40 严格配对门为 FAILED；task8 不可作为 T 对 TF 的配对步数证据。task3 仅证明“离线高通过率不足以保证该局成功”，尚不能定位训练分布或接受规则的单一因果。
- [两阶段六组训练终态](../../../投机采样/results/runtime/p0-target-trace-draft-finetune-v1-20261015-r1/FINAL-RESULT.json)：原 2,000 更新后再做 500 更新、target FIT 与旧 mixed same-query FIT 50/50、两架构各三 seed。TF 的 CAL 中位数 **95.0425%→95.0593%**，MLP **95.4111%→95.4567%**；旧 same-query FIT 为 TF **80.9713%→81.1410%**、MLP **81.4430%→81.6092%**。这些是很小的离线改善，最终文件明确 `deployed=false`，不能称 task3 已修复。旧 same-query 共 14,997 行来自 A_off MLP51 与历史 TF64，而**不是这次新 TF 的 on-policy candidate 查询**；target-only 教师轨迹也不是候选 draft 的访问分布。
- 新 T/两阶段 MLP/两阶段 TF 的 **10 task × init24 × 3 arm = 30 完整局**正在新源下运行。它是观察旧失败后的开发集复核，需等同初始观测/RNG/全程保存、最终 cleanup 与质量门均过后才能解释三臂差异；不能提前引用作新结果，也不能把共享 GPU 墙钟作正式加速比。[恢复计划](RESUME-FRESH30-PLAN-20261015.md)。

## 最小、可复现的下一步

1. **先读取新 30 局终态**，保持旧 40 FAILED 原样。先查三臂真实初始像素、sim/controller、CPU/CUDA RNG 与任务/初态配对；只在严格相等的组内比较成功、双方成功时环境步数 P50/P90/max、失败按 800 步罚分，并列报接受深度、target 调用和 chunk 数。10 对只做工程筛选，不宣称几百分点的总体成功率非劣。
2. 若 TF 仍有配对失败，先对失败 task 及相近成功 task 的**实际新 TF 候选树查询**归档：状态、σ、两 parent/head、target 同状态同条件速度、接受位置、最终动作/观测。按完整 episode 切分新的 FIT/CAL，并保留未参与选模的初态；只训练合法 FIT。旧 14,997 行可作独立数据来源对照，不能重命名为新 TF on-policy。这样验证 DAgger 启发的访问分布假说，但不预设有效。
3. 固定 target、K2/B2 bonus、τ、prefix selector、horizon 和硬件；同预算、同数据切分、至少三 seed 比较“现有 target-only”“旧 same-query 混合”“新增 TF 访问态教师标签”，MLP96 作容量匹配架构对照。只用 CAL 定 checkpoint/阈值，冻结后做新初态完整闭环；如果质量门仍不过，保留 A_off 或原 target，先报告失败原因。按现有门 TF 成功数须不低于同批 T 与 matched MLP，双方成功步数比 P50≤1.10、P90≤1.20、max≤1.30。只有达标后才做独占 GPU 的完整 chunk 正式计时；不在当前证据下扩全 LIBERO100、RoboDojo 或真机。

## CVPR 2027 官方日程复核

[官方 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)和[CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)分别列出 **paper registration/enrollment 2026-11-10 AoE、正文 11-16 AoE、supplement 11-23 AoE**。未另列一个独立于 registration 的 abstract 日期；[CVPR 2027 作者 profile 指南](https://cvpr.thecvf.com/Conferences/2027/CompleteYourORProfile)称它为 abstract submission deadline，并规定截止后不能增删作者，资料不全可能 desk reject。按 AoE=UTC−12 的日期末 `23:59` 换算，北京时间分别为 **11-11 19:59、11-17 19:59、11-24 19:59**；正文与补充的具体 `23:59` 是日期末规划换算，实际提交仍以 OpenReview 精确显示为准。CFP 明确无机构邮箱的新 OpenReview profile 审核**可达两周**；所有合作者现在就应完成注册、可见性、邮箱和冲突信息，最迟不要等到 11-01 老师稿后才启动。项目老师稿目标仍是 **11-01 18:00 北京时间**，应交能审阅的主要结论及负例，而不是把正在跑的质量实验预写成成功。
