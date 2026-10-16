# 短/长KV × 单/双需求：最小固定负载成本对照方案

日期：2026-10-11T16:58:36+08:00（Asia/Shanghai）。路线：**投机采样 action-only，非tile**。

**当前只完成CPU规划和已有输入核查；GPU未启动、预算未确认、新重放包装未实现/未通过工程CPU测试。暂不训练。**

## 1. 本关要回答什么

上一关32对阶段/形状/需求可比请求的action为7.759→9.771秒（+25.93%）；短KV较快、2160长KV明显较慢。但两条闭环轨迹的query数值不同，且受双worker并发影响。本关改用**每个fixture固定输入、逻辑KV、RNG和模型配置，分别改变算法及并发需求**：

1. 同一个固定负载上，当前T与A_off谁更便宜？
2. 在相同fixture/算法/两模型驻留条件下，单请求与双请求并发的成本差异多大？
3. 成本额外项主要在target验算，还是构树/pack/score/route等？

这里短与长是两个真实历史位置的fixture，frame/历史数值也不同，**不能把差异唯一归为KV长度**。可以控制同fixture内的算法/需求变量；不能外推完整任务成功率、环境步数或自然E2E，不能当tile算子结果。

## 2. 最少需要几局：建议0局新完整episode

旧6局job实际保存了`server_output/real/*/obs_data_{frame}.pt`、`actions_{frame}.pt`、`latents_{frame}.pt`，不是只有shape/hash。CPU已逐文件检查T_ON两worker chunk0–13的**84份产物**：28观测窗口、28全30通道动作、28视频latent；两相机uint8原始窗口共440帧，tensor为CPU映射BF16有限值，SHA对应旧保护manifest。

因此推荐：

- **新增完整LIBERO episode：0；新增模型闭环episode：0。**不把固定chunk调用称为“局”。
- **最多2次EGL初态初始化尝试**：同task0/init3、init4，各复用受控seed20260922、原初始化顺序和5步稳定动作；只重建初态RGB/streaming-VAE起点，不执行policy环境动作。两独立worker同时初始化，并验证NVIDIA EGL/provenance。
- 初态20测量数组、client RNG、双RGB、XML/init指纹必须精确匹配旧参考。若失败、产物缺失或重放不成立，就停止，**不自动改成2/4/8新完整局、不重试**。
- 保存的`actions_*.pt`是归一化动作，不直接冒充线上cache实际输入。重建采用冻结源码的`postprocess_action→preprocess_action/_compute_kv_cache`数值路径；仅当单独证明等价时才能使用已有normalized replay shortcut。不能静默省掉round-trip。
- prefix顺序保留原video terminal prime、观测编码和video/action cache更新；原始slot位置/容量/mask/id/is_pred保留。旧文件未保存全KV和server RNG，**不声称本轮CPU已证明与旧线上全hidden KV逐字节相同**；新生成的固定fixture须经GPU结构/恢复关卡后才能测量。

来源：[旧6局job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)；[上一关账本](COST-LEDGER-CPU-READOUT-20261011.md)。本轮核查：[产物inventory](../../../review/OMISSIONS.md)、[候选fixture](../../../review/OMISSIONS.md)、[CPU规划输入验证](../../../review/OMISSIONS.md)。

## 3. 四种负载与固定条件

| 条件 | 原始位置/实际action有效KV | 请求需求 | 模型驻留与分配 |
|---|---|---|---|
| S1 | chunk1 / frame4 / KV288 | 1 | primary执行，另一模型驻留但空闲 |
| S2 | 同一短fixture，KV288 | 2 | 两worker同时跑同fixture、同算法 |
| L1 | chunk14 / frame56 / KV2160 | 1 | primary执行，另一模型驻留但空闲 |
| L2 | 同一长fixture，KV2160 | 2 | 两worker同时跑同fixture、同算法 |

- 每条件使用来自T_ON init3和init4的两份fixture，合计**4份真实历史fixture**。init3主比较worker0，init4主比较worker1；双需求的peer结果是相同输入辅助重复，不算新独立状态/episode。
- 选chunk1而非chunk0，避开初帧obs_encode/零条件；与chunk14的action输入shape同为`[2,30,4,4,1]`，action query16 tokens。实际KV288/2160来自30-block sidecar，不能用pool容量冒充有效长度。两点是在观察成本结果之前固定的方案位置，不是拟合的在线阈值。
- 双需求时**两worker使用同一fixture副本、同算法和RNG**，而不是init3/4不同接受深度互相争用；减少早结束产生single-demand尾部的混淆。T/T与A_off/A_off分开，不混T/A竞争，不把T/A打成大batch。
- 两模型始终驻留；所有恢复/H2D/模式切换/输出落盘在测量窗口之外，各自单列时长。idle peer不做预填充、复制、异步写盘或额外GPU工作。模型/CPU offload/device位置在单双需求间不改。
- 固定Long target及现有A_off.pt、torch attention/original cache/CFG2、global selector、阈值0.007839680526450805、50 action更新＋terminal、20 video更新＋terminal。不改target/draft参数、cache容量、树宽、bonus、seed来源或接受阈值。A_off实际checkpoint以旧job.plan.A_off_checkpoint为准，不误用frozen_reference的历史S0字段。
- 每次调用恢复相同的请求开始cache/valid token与slot元数据、frame、prompt embeddings、调度配置及Python/NumPy/Torch CPU/CUDA RNG；预先加载并冻结predictor。调用原公共`infer/_infer`，不复制另写T denoise loop。公共mode仅在T与普通A_off之间任务内切换，不落为公共默认。
- 同fixture T/A在相同初始噪声/历史起点运行，但算法中间状态和最终action**不要求相等**；video路径应相等，且各算法自身跨恢复/开关/单双需求重复必须精确复现。fixture先建立新的固定RNG合同，不冒充旧未保存server RNG的回放。
- 新固定请求seed预定20260922（加载predictor后设定并保存各RNG）；四fixture请求noise维度相同，固定相同随机生成起点。prefix沿用旧受控初始化合同，两种seed用途分别记录，不声称找回旧中途server RNG。
- snapshot保存有效K/V及**原slot位置和完整元数据**，恢复到原容量，不compact/重排。`torch.empty`无效槽不能要求有限或逐字节相同；只检查有效/实际读取的值，不能用无效槽垃圾值制造FAIL或“全hidden相同”结论。

## 4. 调用数：两块反序，而非更多环境局

每个fixture×需求条件的主测顺序预定为：

```
块1：T → A_off → A_off → T
块2：A_off → T → T → A_off
```

每算法4次主测，两个init分别报告。两块counterbalance只是控制短期漂移，重复请求不视为独立episode。短/长/单/双条件的执行顺序在CPU预检时锁定；分开报告块1/块2，不能临时择优换顺序。

建议四格预定顺序S1→L2→L1→S2；每格块1按init3→init4、块2反序init4→init3。此顺序先写入未来任务合同，不按观察到的速度临时调整。所有warmup/诊断必须先过G1，主测中没有临时增加warmup的许可。

| 用途 | primary/单请求基准数 | 双需求peer数 | 实际完整固定chunk调用 |
|---|---:|---:|---:|
| 主测：4 fixture×2需求×8次 | 64 | 32 | 96 |
| 不计时参考/包装warmup：每算法每fixture每需求2次 | 32 | 16 | 48 |
| profile-ON诊断：每算法每fixture每需求1次 | 16 | 8 | 24 |
| **全部上限** | **112** | **56** | **168** |

实际独立调度slot为112（双需求一次slot包含两次调用）。prefix每worker重建chunk0–13，估计总84次transformer forward（每prefix 1 video terminal prime＋2 cache forward），不是84个policy或episode。reset/prefill/restore/copy/I/O另记，不能混入主action median。

实现不得超过168个固定chunk调用、2次初始化尝试；任何启动失败也消耗该尝试额度，不自动追加调用/补样本/加第三块。失败或噪声过大只报告FAIL/INCONCLUSIVE，等待下一授权。

## 5. 关卡与通过标准

### G0：CPU工程预检（确认预算和实现范围后才做）

新增仅本任务默认关闭的snapshot/replay/计时包装、监督器及独立比较器，不改公共模型/server/client或旧8局成本入口。CPU假模型/小fixture测试：默认不能启动、预算/范围/provenance检查、恢复槽位与所有必要状态、单/双gate真实进程协议、乱序/缺产物/错shape/RNG/超时故障注入、反序预算、缺timer不补0、嵌套phase不重复求和、异常清理。明确CPU实现者和另一次只读比较/审计步骤；项目self模式，不擅自创建subagent。

CPU规划输入PASS **不是** G0工程测试PASS。当前新包装不存在，不可执行该JSON。旧8局guard保持原样，不能填PASS绕过或改其cap后启动本任务。

### G1：资源与fixture恢复关卡（GPU开始后）

- 新鲜独占GPU/共享锁、两server及master/coordinator端口、磁盘≥40GiB、CPU RAM额外余量目标≥10GiB、冻结source/9target size-mtime、EGL live audit/provenance。当前CPU磁盘约60GiB只是当时值，不代表GPU资源已可用。
- 两次初态精确匹配；真实历史窗口、prompt/frame与正确的cache更新顺序一致；30 blocks实际action有效KV达到288/2160，CFG普通batch2与verifier batch14、query16，cache layout/capacity不改。
- 每fixture恢复后的**全部有效K/V、slot元数据、输入/RNG/配置合同hash精确一致**；必须覆盖原始cache的暂存/恢复与terminal写入，不只验证block0或最终action。
- 每算法2次warmup：先无新计时/记录包装的原路径参考，再恢复同fixture开启新主测包装；同算法完整30通道action/video形状、dtype、逐字节值，以及终止后的有效cache、RNG和调用/调度结果一致。
- 1次profile-ON诊断与其profile-OFF参考也须逐字节一致。不同算法action不强制相同；同fixture视频应精确相同。未保存的隐藏状态不能补假hash或放宽成allclose。
- 任一restore/输出/phase/预算检查不通过就停止，不进入主测。不能从之前自然6局PASS推定新snapshot/恢复包装也PASS。

### G2：有效成本测量标准（与“算法更快”分开）

主测**关闭公共phase profile和中间CUDA同步**，只在action整体入口/出口及policy整体端点同步、取墙钟；保留轻量Python调用/调度计数，不在主窗口中复制GPU张量、全block抓QKV、序列化或落盘。不把全部包装叫完全零开销或clean episode E2E。

双worker先在恢复完成后同cohort开始policy，视频完成后再在action入口放行；action同步/门等待放在action计时之外并单列。后者仍位于完整policy墙钟之内，不能把含门等待的policy当自然调度E2E；primary action与整policy分别报告，不能以删除等待美化整请求成本。

- 主action完整包含构树、draft、candidate pack、target/verifier、score/selector/route、50个提交更新和terminal；video/policy墙钟另列。诊断ON的嵌套分解与主测OFF**不同计时口径，不跨lane相减当纯成本**。
- T实际50普通action forward＋1terminal，21 video forward；A_off的实际verifier/ordinary/terminal调用与depth0/1/2和50提交更新对账。每fixture每算法的调用/深度跨主测重复固定，不能只固定50调度步却放任接受路径变化。
- 双需求主action区间：两worker真实区间交集/各自action时长均≥90%。这是建议的测量有效性门，**不是GPU kernel重叠证明**。若不足，停止该关并标负载不成立，不加dummy请求、不强补peer或掩盖尾部。
- primary各fixture/算法/需求4次主测均完整，median>0、finite；MAD/median≤7%。超阈值为INCONCLUSIVE，不自动扩样/调整容忍。保留全部原始值、异常和块间漂移。
- 冻结模型/源码/旧结果/protected/raw records不变，任务进程与锁正确收尾，所有预算不超。按matrix逐格记录，不只报告平均快的格。

### G3：成本结果判读（负结果也是进展，不是完整性失败）

每fixture、每需求独立计算`r=median(A_off_action)/median(T_action)`，也报告whole policy/video、实际调用与CFG行单位、块1/块2，以及两个init。相同fixture比较单双需求比值。peer附录不当独立证据加倍。

- **候选加速信号**：r<0.95，两个init及两顺序块方向一致，G1/G2有效性均通过。
- **候选成本退化信号**：r>1.05，两个init及两块方向一致；不是因“慢”停止其余有效格，也不自动改树宽/阈值/训练。
- 0.95–1.05按基本持平处理；方向混合、噪声超限或负载不成立标INCONCLUSIVE。±5%只是预算确认前提出的pilot成本判读带，**不是任务成功率/步数非劣界限或原verifier阈值调整**；本关不作置信区间/统计显著性宣称。
- 若单、双都长KV慢：支持进一步调查当前验算/构树成本预算；若只有双需求慢：支持并发争用/负载路径为优先解释，但仍不定位具体kernel/HBM。若成本可回本，再单独规划profile-off自然E2E；任何模式切换/窄树方案必须重新验任务质量。

## 6. 时间、安全预算和实现边界（均待确认）

| 阶段 | 预计时间 | 当前状态 |
|---|---:|---|
| 任务专用包装/监督器/比较器实现及CPU预检 | 45–90分钟 | 未实现，待确认范围 |
| GPU两模型启动/初始化/prefix/fixture关卡＋168调用上限 | **35–55分钟** | NOT_RUN，预算未批准 |
| CPU独立复核、HW结果与下一步 | 10–20分钟 | GPU结果后 |
| 确认后的总流程 | **约90–165分钟** | 不含GPU排队，非保证 |

**建议GPU阶段90分钟硬安全上限**，从首个任务拥有的GPU/模型/EGL进程启动计时，含启动、reset/prefill、恢复与诊断，不能只算主测或停表等CPU。到限保存incomplete/failed并清理本任务PID/锁，不截短算PASS，不重试，不扩局/加块。GPU按历史T/A_off policy10–20秒及112个调度slot估算，单双需求/恢复/I/O可能改变耗时，不做保证。

有效K/V snapshot预计约2.97GiB（4fixture请求开始有效history；不含元数据/RNG/embeddings），保留原容量GPU池，不复制整模型权重；准备至少10GiB额外CPU余量与40GiB空闲磁盘。实际snapshot体积、CPU/RAM/GPU峰值超过预算则停止或另确认，不删除旧结果腾空间。

建议批准范围必须同时包含：**0新完整episode＋最多2次EGL初态初始化＋最多168固定chunk调用＋90分钟GPU阶段上限＋仅任务专用默认关闭snapshot/replay/监督器/CPU比较器实现**。不含教师训练数据、RL、联训、公共模型改造、服务重启或系统依赖安装；不自动跑旧8局ABBA或完整自然E2E。

## 7. 本轮已做与未做

- 已做：既有84份产物CPU读取与shape/dtype/finite/hash、440原始相机帧窗口检查、四fixture位置与实际QKV、168调用算术、保护manifest/9target stat、方案与HW增量记录。
- 另有8项规划合同检查（含预算/开关/计时口径/扩范围故障注入）PASS，见[规划合同测试](../../../review/OMISSIONS.md)。它们验证候选方案，不是尚未实现的G0重放工程测试；本轮1559既有文件SHA/9target stat保护，历史tile和protected不改。
- CPU映射读取既有Torch tensor，`CUDA_VISIBLE_DEVICES=''`、`map_location='cpu'`，CUDA未初始化；未import模型/LIBERO/MuJoCo或创建EGL。
- 初版只读检查误将原始相机窗口当latent帧（3/4而非12/16），随后又低估双相机latent空间shape（实际8×16）；仅修正临时审计假设后重跑，两个初始错误日志保留。没有修改旧产物或工程代码。
- **未做**：新snapshot/replay工程实现及G0测试、新GPU/服务/EGL初始化/episode/模型调用、训练/teacher/RL/联训、资源可用性核验、新fixture恢复的GPU证据、任何新性能结论。

## 8. 下一步提醒

**先确认上述预算与任务专用实现范围 → CPU实现并预检 → 新鲜资源/EGL门 → fixture恢复/不扰动门 → 四格成本主测 → CPU复核/HW汇总。**每关结束汇报本步与下一步；完整性失败停止，负成本结果如实记录。只有后续另批准才走自然E2E、成本感知算法或冻结target真实状态训练，当前不做RL/KL或大小模型联训。
