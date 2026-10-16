# 2026-10-15 早间 1→2→3：投机侧结构与全 Long TRAIN 数据

时间均为北京时间。本文是研究记录，不是论文中文稿。当前线上研究候选仍是 A_off；独立 tile Transformer 代码与 HW、撤回的算子融合和现用配置均未修改。

## 问题与完成情况

本轮按用户约定先登记历史 task/init/seed 数据用途，再为 task2–9 的 init17 补采同查询 teacher 标签，最后固定旧训练与筛选协议比较 TF64 和参数量相近的 MLP96 各三种随机种子。训练后的准入只代表旧 DEV 的离线候选资格；新任务 DEV、原生 GPU 延迟、闭环成功率和 trace 长度均是独立的下一关。

1. **用途登记完成。** [逐组合台账](../../../review/OMISSIONS.md)覆盖 500 个标准 task/init 组合，[汇总](../../../review/OMISSIONS.md)登记了历史 45 次训练与三种种子的使用路径。所有 500 个组合只有历史完成元数据，不等于 500 次独立试验；目前认证的独立最终 TEST 数为 0。历史 teacher 权重当时未逐字节哈希，不能事后推断与现存权重完全相同；本轮九个现存 teacher 权重另有 [内容哈希](../../../review/OMISSIONS.md)。
2. **新增八条完整 TRAIN episode。** task2–9 各取 init17，四个顺序双 EGL wave 全部以 v4 PASS 结束，原始身份、标签、文件 SHA、累计成本和自有进程清理均有独立终态复核。task6 到 horizon 后失败，仍按预登记规则保留完整轨迹，不做成功样本筛选。新增 9,120 行，加旧 task0/1 init17 的 1,971 行，总辅助池 11,091 行、3,697 棵树且每树三节点。八条中七条采集任务成功、一条失败；这只是 TRAIN 轨迹来源，不是新算法的成功率。四波 GPU 协议时间合计约 2,575 秒，累计账本到 42,886.176 秒、200,904 charged target intents、50 model launches，恰达本轮模型启动上限。四波终态分别见 [wave0](../../../review/OMISSIONS.md)、[wave1](../../../review/OMISSIONS.md)、[wave2](../../../review/OMISSIONS.md)、[wave3](../../../review/OMISSIONS.md)。共享 GPU 采集不用于性能比较。
3. **扩展训练输入独立核验通过。** [CPU 训练器](../../../review/OMISSIONS.md)准备出精确 plan SHA `6a18a8e71256405fc7813a175d10e0a67fbd5fcd7d5a788c7b7e637452e47b4d`。交叉审计逐文件重哈希 4,300 项，确认四波链、旧 1,971 行与当前/DEV 缓存原始字节、旧 TF_A 三 seed 权重与全 DEV1890 预测/六门均一致；辅助池对旧 current3906、DEV1890 的 227 维 FP32 输入逐字重复均为 0，辅助池内部重复亦为 0。见 [独立交叉审计](../../../review/OMISSIONS.md)与[重合审计](../../../review/OMISSIONS.md)。这种精确输入重合审计不证明图像/KV 相等，也不建立独立 TEST。

## 冻结结构与训练合同

TF64 输入是当前动作状态 112 维、上一去噪方向 112 维、`sigma/next_sigma/timestep÷1000` 三标量，共 227 维。按七个动作通道、十六个位置重排为 16 个 17 维 token，经 `Linear(17,64)`、可学习位置嵌入、一个 pre-norm TransformerEncoderLayer（4 heads、FFN 256、GELU、dropout 0）、`Linear(64,14)`，每位置拆成 7 维 center 和 7 维 delta，形成 `center±delta` 两个候选方向。TF64 为 53,070 参数；对照 MLP96 为 52,928；现用 A_off 为 25,928。源实现见 [models.py](../../../投机采样/experiments/draft_architecture_v1/models.py)。

两结构沿用原 current3906、归一化、初始化与三种 seed `20260922/23/24`；每组 AdamW、学习率 `1e-4`、weight decay `.01`、1000 更新；每次 512 current +128 新辅助样本，损失 `current_C + 0.25 aux_DATA`，不加辅助 KD。旧 task0/1 init8 的 1,890 行 DEV 和原六门只用于兼容历史的离线准入；新任务上的训练池诊断不参与选模。旧 TF_A 三 seed 对照在原样重放且逐 tensor 一致后复用。训练中的模型不得改动当前 A_off preset。

## 六组训练与独立后审

六组 CPU 训练均完成 1,000 更新，退出码 0，无 `PARTIAL-RESULT.json`，CUDA 未初始化，准备加训练共 304.174 秒。[冻结最终结果](../../../review/OMISSIONS.md) SHA256 为 `a959149227a03e3f5c94ac0ee2496d95c229c62555c44e96910f4d30a83aec72`。另一套[独立后审](../../../review/OMISSIONS.md) SHA256 为 `35d7ce6f485b3f66d743189b8c2ea504d712df78cfd73e3e580db70c41c3b87b`：逐个重载六 checkpoint，重算旧 DEV1,890 行预测、六门、CPU 真实 scheduler 后继诊断与三种子排序，均与训练报告相同。审计源码在 [posttrain audit.py](../../../review/OMISSIONS.md)，SHA256 `d60638741b364785c38e06419558ad84f5c62652343c4aa5d3bc12690328cd15`。

| 结构 | 三 seed 根节点通过数（旧 A_off：527/630） | 单 seed 六门通过 | 三 seed 中位主要 margin | 结论 |
|---|---:|---:|---|---|
| TF64，11,091 扩展辅助行 | 530 / 532 / 528 | 0/3 | 根通过 +3；task0 根通过 −3；旧通过根节点保留率超标 0.8975 个百分点 | 不准入 |
| MLP96，同一数据与更新数 | 510 / 512 / 517 | 0/3 | 根通过 −15；task0 −8；task1 −7；旧通过根节点保留率超标 2.4156 个百分点 | 不准入 |

TF64 三 seed 的 task0 根节点通过数分别为 262/264/260，旧 A_off 为 265；旧已通过根节点转失败为 10/9/13 个，原保留门允许的比例上限为 1%。TF64 扩展数据相对逐 tensor 重放的旧 TF_A 三 seed，根通过数从 525/531/525 到 530/532/528，显示**旧 DEV 局部根指标小幅提高**，但没有一个 seed 满足原六门。MLP96 的根通过数、task0/task1 及根后继误差均低于原门。独立计算给出 `selected_offline=null`、`KEEP_A_OFF_NO_OFFLINE_ELIGIBLE_CHECKPOINT`；现用权重与接受器配置未替换。训练池各任务上的指标只做 TRAIN 诊断，不能推出新任务成功率或轨迹长度。

## 代码入口与在线投机路径

[代码导航](../../../投机采样/docs/CODE-MAP-20261014.md)及[当前配置](../../../投机采样/configs/current_speculative_candidate.json)指向当前 A_off、K2/B2、bonus、共享 KV、`prefix_then_cost`，阈值为 `0.007839680526450805`。现用权重是 `投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt`；TF64 仍是隔离训练结构，不是现用权重。

服务器每个动作去噪 chunk 共执行原定 50 次 scheduler 更新。先用一次大模型方向建立上一方向；之后小模型从当前紧凑 112 维状态和上一方向各给两个连续候选，构成深两层、两分支的树（根 1、内部 2、叶 4）。七个候选状态批量交给冻结大模型验证；CFG2 下物理批为 14 行。每条边的局部误差是 `|Δsigma|·||draft direction−target direction||₂`，与固定阈值逐边比较；最长可接受连续前缀优先，同深度比较已接受边代价。只提交通过的前缀；首条或第二条失败时沿第一失败边使用大模型真实方向。两条都通过时，bonus 用叶节点已验证的大模型方向再执行一次真实 scheduler 更新。最终 `t=0` cache forward 仍由大模型执行。代码分别在 [draft/构树与打分](../../../投机采样/scripts/p0_batched_verify.py)、[前缀选择](../../../投机采样/experiments/prefix_acceptance_v1/selection.py)、[服务器动作更新](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)。

## 是否可开始 MegaTransformer 算子线

可以在隔离目录以冻结 A_off 为算法基线开展算子研究；投机侧的论文质量证据尚未收口。既有短/中/长 KV 的完整 policy chunk 对原 T 分别 1.602×/1.544×/1.500×，45 个请求、18/18 正式配对更快；这些计时不含 reset、环境步进或模型加载。真实 verifier profile 中 cross-attention 占 GPU kernel 时长 54.40%–62.22%，故首轮可研究请求作用域内静态 CFG14 文本 K/V 的生命周期与复用；旧 prompt-KV 缓存曾让请求变慢，不可算作新收益。数值与性能门、预算和原 tile 隔离边界见 [算子启动审查](../../../review/OMISSIONS.md)。该首轮尚未实现新算子，也尚无新增性能结果。

## 下一步

由于本轮无合格 checkpoint，不启动新权重的 GPU native 计时或闭环质量；先保留 A_off。下一次投机侧实验应预登记并只在 CPU 上用**已有数据**做根节点失分诊断：逐 task、sigma、初态和旧通过→失败样本定位，再比较任务均衡辅助抽样与显式旧通过根节点保留损失的配对消融；仍用同三 seed、冻结旧六门，并另外指定未参与选择的新任务 DEV。现有辅助池中 task6 一条失败轨迹占 2,745/11,091 行（24.75%），task0/1 合计 1,971 行（17.77%）；这种数据量差异值得作为消融变量，但本轮没有证明它就是失败原因。若 CPU 消融产生稳定合格候选，再实现兼容服务器的 TF loader，做原生 GPU 数值/完整 chunk 延迟和预登记闭环质量。

闭环须在冻结接受规则下配对原 T、A_off 与新候选的成功率、环境步数分布和长尾、完整任务墙钟。只有足够独立的任务/初态与事先约定的非劣门槛，才能支持 CVPR 的质量声明。标准 500 组合已有历史执行记录，正式 TEST 划分须在用途台账中逐一认证或另建未触及的最终集。当前不要把全 LIBERO100、RoboTwin/RoboDojo 或 Fast-WAM 全规模测试放在结构准入之前；可先完成 Long10 小规模新 DEV，稳定后再扩展第二模型/模拟器。算子线可并行从隔离 block 原理验证开始，等质量和系统两线各自通过后再做组合消融。论文中文方法可在对话中逐段写，按既定 11 月 1 日 18:00 北京时间交老师的时间倒排，不生成本地论文初稿。
