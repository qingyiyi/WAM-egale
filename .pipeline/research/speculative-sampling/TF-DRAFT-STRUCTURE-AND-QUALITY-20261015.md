# TF draft 结构与质量复核（2026-10-15，北京时间）

## 当前结论

当前线上研究候选继续使用 A_off、K2/B2 bonus、`prefix_then_cost` 阈值 `0.007839680526450805` 和已有共享 KV / O1–O3。TF64 是可运行接口已验证、但离线结构门槛未通过的研究候选，**尚不能替换 A_off**。本次两种补充训练方案各做三 seed，仍是 0/3 通过旧六门。LIBERO pilot r7/r8 分别发现隔离入口的预加载动作维度与 O3 双环境开关遗漏，均未产出 A_off/TF 完整配对。r9 修复后通过 CPU 门、完成一条 T/task0/init20（成功，285 环境步），但在用户要求停止小规模试跑后终止并清理；T/task1 未完成，A_off/TF 均未执行完整闭环。因此完整配对、TF 成功率、轨迹长度和 TF 加速比现在仍不得声称。

## 结构与条件输入

`投机采样/experiments/draft_architecture_v1/models.py` 的 TF64 接受 112 维当前动作状态、112 维上一步方向、`sigma/next_sigma/timestep/1000` 三个标量。每个 16 个动作位置构成一个 17 维 token（7 个状态通道、7 个上一方向通道、3 个时间标量），经 `Linear(17,64)`、可学习位置向量、一层 pre-norm、4 头 self-attention、FFN 256、`Linear(64,14)` 得到双候选方向；参数 53,070。它没有 cross-attention，也没有读取文本 prompt、T5 向量、图像或视频 latent。现用 A_off 是 25,928 参数 MLP，匹配容量的 MLP96 是 52,928 参数。网络结构是否适合论文，不应代替质量准入。

原 WAM server 在 `_reset` 时编码并持有形状 `[1,512,4096]` 的 T5 prompt embedding，target 已使用语言条件。当前训练数据保存 10 个任务的原始 prompt 字符串，但没有 T5 embedding、图像或 target 中间 KV。已记录[文本 cross-attention 的最小结构和对照设计](../../../review/OMISSIONS.md)：以动作 token 作 Q，冻结 T5 的有效文本 token 经投影后作 K/V，按 reset 缓存文本 K/V，维持同样双候选头与接受器。此方案是待检验假设，新增约 28 万参数，不能用现有 TF64 的结果冒充它的效果。视觉条件版还须新采集与 teacher 节点对齐的观测/latent。

当前 `FlowMatchScheduler` 直接计算 `timesteps = sigmas * 1000`。旧 DEV 1,890 行实测 `max |timestep−1000×sigma| = 2.002716064453125e−5`，仅为 FP32 舍入；固定 50 步时 `next_sigma` 也由当前 `sigma` 决定。三项一起输入沿用了原 227 维特征与已训权重的接口，**没有实验支持“三项都必要”**。`next_sigma` 可以显式表达本步步长，未来改变调度或步数时未必冗余；应在同一数据、seed 和旧六门下比较 `sigma,next_sigma,timestep/1000` 与 `sigma,Δsigma`，而不是直接删已有权重的输入列。

## 离线训练结果

所有补充单元仅用封存的 Long10 辅助 TRAIN 和旧 task0/1 init8 DEV，TF64 结构、当前 TRAIN 3,906 行、每 seed 1,000 更新、512 current + 128 aux 和原六门固定。旧 DEV 已反复用于研究，是开发集而非独立 TEST。

| 条件 | seed22/23/24 根通过数（A_off 为 527/630） | task0 根通过数（A_off 为 265/311） | 旧通过根转失败数 | 全六门通过 |
|---|---:|---:|---:|---:|
| 早间 TF64 扩展辅助 | 530 / 532 / 528 | 262 / 264 / 260 | 10 / 9 / 13 | 0/3 |
| 本次 task 均衡辅助抽样 | 529 / 531 / 526 | 261 / 263 / 260 | 12 / 10 / 15 | 0/3 |
| 上项再加 2× current 根保留 KD | 528 / 527 / 524 | 262 / 260 / 259 | 11 / 12 / 16 | 0/3 |

早间匹配容量 MLP96 为根 510/512/517，旧六门也是 0/3。本次两种新训练都没有稳定修复 task0 及原有通过节点保留；单看总根通过数会掩盖这两项损失。训练原始终态在 `/var/tmp/cxz-wam-quality-20261014/p0-tf-balanced-retention-cpu-v1-20261015-r1/FINAL-RESULT.json`，状态 `PASS_SIX_CPU_TRAINING_EXECUTIONS`、`eligible_conditions=[]`、CUDA 未初始化。它只表示六个训练单元执行完成，不表示准入通过。早间对照见 [Long10 结果](MORNING-LONG10-123-RESULT-20261015.md)。

## LIBERO 配对 pilot 的边界与状态

隔离实验入口为 `投机采样/experiments/tf_libero_pilot_v1/`。TF adapter 的 CPU 精确对照在两 seed × 12 个旧 DEV 行 × FP32/BF16 通过；服务端导入顺序的 CPU 预检将固定 v4 verifier adapter 与独立 TF adapter 区分开，且未初始化 CUDA。实验预登记原 T、A_off、TF seed22/23 四臂，各 task0/1 init20 一个完整 horizon800 episode，比较成功、环境步数、chunk、target 调用和接受深度；主 TF 为 seed22，seed23 作敏感性。全部只是复用条件上的诊断，不是统计非劣性或正式速度试验。

原始终态逐次保留：

- r4 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r4/FAILED.json`：首服务导入时固定 v4 adapter 路径冲突；0 episode，1 次模型启动，推理前失败。
- r5 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r5/FAILED.json`：旧验证器入口的同名 `task_contract` 冲突；0 episode，1 次模型启动，推理前失败。修复后配置判断与旧入口 AST 完全一致。
- r6 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r6/FAILED.json`：外部 PID 3438178 在模型加载期占据约 37.7 GiB，runtime 空闲显存跌破 2 GiB，保护逻辑中止并清理自有进程；0 episode，1 次模型启动。此项不是 TF 质量失败。
- r7 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r7/FAILED.json`：两条 T 完整成功，A_off/task0 在 READY 前因隔离入口提前调用原生 loader，而生产 server 的 `action_per_frame` 到首次 `_reset` 才建立；0 条 A_off/TF 闭环。两条 T 后因终态失败按协议全额记账，累计 720 秒 GPU、3 次模型启动、11,250 次保守 target intent；owned 清理通过。
- r8 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r8/FAILED.json`：T/task0 与 T/task1 完整成功；A_off/task0 已 READY，但首个 verifier tree 因隔离 worker 未传 `PREFIX_ACCEPT_ENABLED=1` 与 `PREFIX_ACCEPT_GPU_APPROVED=1` 而失败，未形成 A_off/TF 完整局。此为实验入口缺陷，不是算法质量结果。
- r9 `/var/tmp/cxz-wam-quality-20261014/p0-tf-libero-pilot-v1-20261015-r9/FAILED.json`：补齐 O3 双开关，CPU gate 通过目标权重、24 条 TF 数值和 8 条 worker 环境检查。第 1 条 T/task0/init20 完整成功（285 步、19 chunks、1404 target intents）；用户要求不做小规模试跑后收到 `STOP_REQUESTED` 并清理全部自有 GPU 进程。仅保留该 T 局为非正式历史证据；A_off、TF 均无新闭环结论。

## 论文与下一步判断

在现有证据下，Transformer 结构可以继续作为方法候选，但不能仅因为它比 MLP 容易写而宣布替代。按用户最新要求，不再补 r9 式小规模试跑；A_off 作为今日全量采集与评估冻结候选。若继续探索 TF，应在完整训练数据上做原 TF64、参数量匹配 MLP、pooled 文本和文本 cross-attention 的公平对照，每个方案至少三 seed 并通过旧六门，再进入正式完整闭环评估；冻结 T5 embedding 可按相同 tokenizer 和 prompt 重建并核对。正式论文需要独立质量集、成功率与环境步数长尾，以及独占 GPU 下完整 policy chunk/端到端时延；目前已有 A_off 的短/中/长 KV 单 chunk 相对原 T 1.602×/1.544×/1.500×，不含 reset/env/加载，不能转写为 TF 或完整任务收益。论文表、图、样本口径与 AutoDL/本机分工见[全量评测矩阵](FULL-SCALE-EVALUATION-MATRIX-20261015.md)。

目前可与质量线并行写中文方法草稿（对话逐段，不在仓库写论文初稿），和在隔离目录中设计 verifier 的 MegaTransformer 数值/性能门。原 tile Transformer 方向、已撤回融合、A_off 部署 preset 均不动。计划按 2026-11-01 18:00 北京时间给老师看稿倒排；当前最大的论文本体缺口是投机侧任务质量的独立证据，不是网络名字。
