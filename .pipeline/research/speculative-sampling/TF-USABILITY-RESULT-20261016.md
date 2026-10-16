# TF64 可用性有限验证：2026-10-16 终态

本报告接续[夜间结果](OVERNIGHT-RESULT-20261016.md)，对应用户晨间授权“本机和 AutoDL 都可继续用，优先确定 TF 是否可用”。这轮 GPU 工作已结束。AutoDL 已交还用户关机，后续不再 SSH；实例是否已关闭未核实。

## 结论与实际选择

保留 TF64，固定 TF_CONT 最后一次更新的权重，作为下一轮扩大质量验证的候选。新的 20 个 Long10 task/init 组中，TF_CONT 与 MLP 均 20/20 成功，target 和旧 TF child 均 19/20；TF_CONT/T 双方成功的 19 局，环境步数比 P50/P90/max 为 **1.004184 / 1.032907 / 1.172996**，通过原先六项数值筛选。

这支持 TF64 结构继续研究。当前不能签发原严格质量证书，不能把它设为生产默认：全 60 组初始相机仅 56 组逐位一致，TF_CONT/T 主比较是 19/20；远端还存在原终态元数据 FAILED，冻结聚合器真实返回 PARTIAL。固定恢复病例又显示闭环重复性有限。因此，这轮没有证明 TF 优于 MLP、没有证明增加训练修复了历史困难病例，也没有建立独立 TEST 或 3pp 非劣结论。

工程质量筛选、独立 TEST 完成、非劣建立、正式性能许可四项仍为 false。旧 A_off 默认、原接受器及 TF 正式 45chunk 入口的限制均保持。

## 实际做了什么

1. 事前固定主候选 TF_CONT，控制组 T、TF_CHILD、MLP_CHILD。Long10 每任务两组 DEV：本机 init33、AutoDL init34，四臂共 80 条完整 episode，horizon800，无截断、无新 TEST40–49 解盲。
2. TF_CONT 从固定 child seed20260923 出发，沿原数据配方再训练 500 updates；最后一次更新固定，不挑最佳 checkpoint。CPU 125.154 秒完成，未新增 teacher 采样。训练 batch512 中历史 target256、旧混合池256；normalization、loss、学习率和推理 dtype/device 均保持。优化器重新初始化且沿用原500步抽样安排，不能称为无缝连续的3000步训练。
3. 三个 draft 在各站均完成真实 20237 行 CAL 的 GPU adapter/原 decode BF16 逐位核验。CPU 接受率由95.059%到94.776%只作旧 CAL 描述，未据此判定任务质量。
4. 两站分别一个 GPU worker 跑完整40；本机首次监督进程意外消失，保留原13条完整与中断记录，原因 UNKNOWN。第二次独立会话将同40条全量重跑，继承原累计3小时和两目录3GiB；旧13不混入新80。
5. 保留 AutoDL 首次 EGL 启动失败及第二次终态 FAILED。完成全部原始数据 CPU 复核、独立80条描述重建、固定13条恢复重复性核查，并真实运行冻结的原聚合器。没有再训练新数据版本，没有执行新的 PF/阈值/树结构搜索。

TF64 仍为 53070 参数：16token×17输入，单层 pre-norm、hidden64、4 heads、FFN256、GELU、dropout0、学习位置编码、两个7维残差头。输入 state7、前一去噪方向7（根节点用调用方已有方向，递归节点用父draft方向）、sigma、next sigma、t/1000；无 prompt/image cross-attention。MLP child 为52928参数。这里校正旧快照中“每次都输入前次target方向”的术语；实际树展开见[真实源码](../../../投机采样/scripts/p0_batched_verify.py)557–570行，计算和实验未改。TF_CONT 额外训练500步，MLP没有同轮额外500步，所以本轮不是等更新预算的架构优劣实验。

本轮保持 K2/B2 bonus、7状态×CFG2=B14、prefix_then_cost、共享历史/O3 和阈值0.007839680526450805；BF16模型方向、FP32打分。PF2未启用。新MegaTransformer及文本投影缓存没有接入这轮质量路径。

## 80 条完整 DEV 的独立描述

| 方案 | 本机 init33 | AutoDL init34 | 合计 | 实际环境步总数 | 失败记800的均值 | target调用/chunk |
|---|---:|---:|---:|---:|---:|---:|
| T | 9/10 | 10/10 | 19/20 | 5562 | 278.30 | 73.895 |
| TF_CHILD | 10/10 | 9/10 | 19/20 | 5705 | 285.45 | 43.408 |
| TF_CONT | 10/10 | 10/10 | 20/20 | 5183 | 259.15 | 43.524 |
| MLP_CHILD | 10/10 | 10/10 | 20/20 | 5170 | 258.50 | 43.674 |

target调用包含原计数边界下的 video/action 等调用，不能直接换算速度。共享硬件上的 episode wall 只作诊断，没有生成新的正式单chunk、task或E2E加速比；SXM4与PCIe的时长不混写。

失败均保留：T在本机task8/init33失败，TF_CHILD在AutoDL task1/init34失败；二者实际执行796步，加4个稳定化步到原800上限。失败罚分800与实际步数796分别记账。

| 对照 | 双方成功局数 | 步数比P50 | P90 | max |
|---|---:|---:|---:|---:|
| TF_CHILD/T | 18 | 0.999870 | 1.028740 | 1.054496 |
| TF_CONT/T | 19 | 1.004184 | 1.032907 | 1.172996 |
| MLP_CHILD/T | 19 | 0.996154 | 1.025940 | 1.083916 |

TF_CONT/T配对四格为：共同成功19、T胜/TF败0、TF胜/T败1、双败0。观察成功率差+5pp；事前固定的任务内配对初态bootstrap描述95%区间[0,10]pp，仅描述这20组。TF_CONT/MLP两者20/20使同种bootstrap得到[0,0]，这不表示总体没有不确定性。20组不能证明3pp非劣；也没有多训练seed的完整闭环证据。

## 严格配对、原 FAILED 与恢复重复性

全部60组实际保存的初始物理/控制器、目标CPU/CUDA RNG及reset核验通过。严格相机56/60，四处差异均在task2腕相机、最大幅度1：

| 站点/初态 | 对照 | 变动pixel/channel |
|---|---|---:|
| 本机33 | TF_CHILD/T | 2/6 |
| 本机33 | TF_CONT/T | 1/3 |
| 本机33 | MLP_CHILD/T | 1/3 |
| AutoDL34 | TF_CHILD/T | 1/3 |

这些组保留在完整统计中；不设置事后容差、不删组或将严格子集替代全部20。不能把主TF观察20/20写成20对输入全部逐位相同。

AutoDL原40条已采集、全部任务和自有进程清理完成。最终扫描把位于作业内的历史凭证 `PREDECESSOR-FAILED-ADMISSION.json` 识别为失败标记，故原 `FAILED.json` 保留、原 `RESULT.json` 缺失。它不是新增的 TF_CONT 任务失败，也不能改名、删除或补造RESULT变成PASS。冻结聚合器实际退出2，验证本机完整40，因远端原标记返回 `PARTIAL_OR_INVALID_NO_COMPLETE_AGGREGATE_SELECTION`。另行完整80条 raw 描述不修复原终态或权限。

未来启动已准备独立布局检查器 `tf_usable_admission_v1/check.py`，对实际不合格布局会在GPU前拒绝；历史失败凭证应置于新活跃作业目录之外。该工具尚未接入旧运行入口，本轮不追改原扫描规则。

固定全部13条恢复前已完成病例，与最终r4同名病例比对：初始保存物理13/13、target RNG13/13、初始相机11/13、完整命令10/13、成功结果12/13、步数10/13一致。三条差异：

- T task2/init33：初始图像相同，255→265步，均成功，命令首次差异在索引12/channel4。
- T task8/init33：初始图像不同，旧393步成功→新796步失败，命令从索引0不同。两轮均遵守完整800上限。
- TF_CHILD task2/init33：初始图像不同，245→239步，均成功。

这是闭环重复性风险的实证，不能把像素差、GPU数值、恢复方式或特定算子直接写成原因。旧13受中断顺序选择，不是新独立样本、不混80、不替换新失败。仅有记录的Python/NumPy RNG hash，没有原对象；没有证明全部外部依赖的RNG覆盖。源码静态审查未发现已证实的额外抽样缺口，external diffusers内部仍UNKNOWN。

## 历史负结果与系统主线

原 child50 的 T48/50、TF44/50，差−8pp，共同成功最大步数比1.40；原fresh30、旧T/A_off100严格FAILED及所有数值负结果继续保留。本轮TF_CONT新20组不能证明历史失败已修复，也不能与旧50混成同一权重70组测试。

系统主线仍是“投机改变形状→可调度多候选→超过最佳B14的新增调度→质量约束下真实收益”。这轮优先质量，没有新增系统集成：H1已测真实Linear M32→224且库已合并、attention每实例Q16；H2同冻结query批处理经济性成立；H3有guarded路线B和共同文本缓存下静态30层单进程pilot，但独立复测、缓存生命周期及cached完整chunk未完成；H4仍缺合格质量证书及新完整闭环性能。详见夜间报告，不将旧倍数相乘或把普通batching命名为新创新。

## 预算、归档与清理

两站40条终态分别计36600、35714次真实target调用。TF训练CPU125.154秒；独立80重建CPU14.793秒、固定13复核CPU2.635秒；root实际原聚合CPU0.083秒。

本机从原r3起累计时钟7815.595秒、远端从原r1起4700.668秒，合计3.47674小时，均低于各3小时/合计6小时帽。此数包含旧失败、重试间隔与等待，是保守记账时钟，不是CUDA活跃时长或性能结果。本机新旧质量目录合计1,923,472,695字节，低于3GiB；AutoDL原40目录约1.418GB，低于3GiB。

本机所有已观察自有worker、监督及独立guard已退出，最终额外核查无信号发送；AutoDL已完成原自有清理且guard终态留档。用户要求可关机后不再SSH，也不声称卡上所有未知进程都属于别人或GPU必为空。未启动新的GPU作业，未购买资源、续租或代关实例。

远端小档本地约65MB，40条actions/latents、初始相机/状态/RNG、协议及终态齐备。664份obs_data约1.352GB保留远端，其值已在关机前由远端CPU实际复核；本地不冒充读取这些值。远端2070个PT、22334个数组的实际核验回执与40条映射已本地归档。没有删除失败原件。

## 下一步默认路径与收缩

1. **先收口评测重复性，再扩大训练。** 用已保存raw定位上述首个分歧的边界，核initial render、完整状态/RNG记录、client/server消息和数值路径；仅原始字段支持的部分下结论。后续如需新增完整重跑，先登记病例、重复数、输入与资源帽。补齐布局预检的实际启动器集成，原FAILED保留。
2. **TF64结构和本次权重暂冻用于扩大验证。** 不再盲目加层、加cross-attn或PF搜索；先核历史困难病例，明确它们是已解盲DEV回归。新独立TEST仍前瞻固定任务/初态manifest；评价成功率、配对胜负及区间、共同成功长度、全任务800罚分和真实成本。候选/系统版本改变不能自动继承旧证书。
3. **系统可并行推进共同强baseline。** 独立复测文本投影缓存、生命周期/内存回收、完整chunk状态正确性及新profile，然后核新增调度相对最佳B14的增量和单位有效提交成本。原tile不改；正式TF45chunk仍等质量许可和本机无争用窗口。
4. 一次新FIT/CAL及TF-on/MLP-on仍HOLD；现有数据不足以证明访问分布缺口，新增训练先有证据。当前equal-capacity MLP保留；论文架构对照需补足相同更新预算，而非据本轮额外500步宣称TF更优。
5. 中文方法可以按冻结的模型/连续去噪树/验证与提交边界逐段写在对话，由用户放飞书。质量结论和系统端到端主表等待实测；不新建本地论文初稿。

默认日历：10/16–18收重复性/协议与强系统baseline；10/21有限DEV收口；10/22质量+系统go/no-go；10/23候选/引擎/独立TEST清单冻结；10/24–26独立TEST；10/27获准的正式计时；10/28主表；**11/01 18:00完整中文稿给老师**。主100对加三个50局baseline共350条的既有估计为PCIe单worker12.5–15.6小时另留1–2小时，长尾可超18小时；不是本轮已启动作业，也不是新增租卡授权。

乐观：10/17重复性边界明确，10/21前质量/系统关键证据完成。常规：沿默认日历执行，第二环境/真机不展开。不利：10/22仍无质量或新B14增量证据就停止加方向，明确负结果与缺口；不放宽旧门、观察TEST后补样直到通过或改变投稿目标。

CVPR2027官方日期本轮已重新核验：注册11/10 AoE、正文11/16 AoE、补充11/23 AoE，即北京时间11/11、11/17、11/24各19:59。官方页面未另列独立abstract截止；10/18检查所有作者OpenReview profile、10/25处理遗留。见[官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)及[投稿要求](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)。

## 原始证据导航

- [本轮登记/状态/预算与root独立交叉核](../../../投机采样/results/runtime/p0-tf-usability-v1-20261016-r1/PLAN.md)：同目录AGGREGATE-PLAN、RAW-STATS-CROSSCHECK、FINAL-RESOURCE-AND-OWNED-CLEANUP、AUTODL-RELEASE-TO-USER。
- [独立80条小结](../../../投机采样/results/runtime/p0-tf-usable-80-raw-descriptive-v1-20261016-r1/SHORT-REPORT.md)：同目录DESCRIPTIVE-REBUILD及SUPPLEMENT，完整数值、来源SHA和远端实际回执映射。
- [冻结原聚合真实PARTIAL](../../../投机采样/results/runtime/p0-tf-usable-aggregate-v1-20261016-r1/AGGREGATE.json)。
- [本机40原RESULT](../../../投机采样/results/runtime/p0-tf-usable-quality-v1-local-i33-20261016-r4/RESULT.json)；[AutoDL40原FAILED](../../../投机采样/results/runtime/p0-tf-usable-quality-v1-autodl-i34-20261016-r2/FAILED.json)，各原目录保留PAIRING/终态/cleanup/clock。
- [固定13条恢复重复性](../../../投机采样/results/runtime/p0-tf-usable-recovery-13-repeatability-v1-20261016-r1/SUMMARY.json)；[最终口径独审增量](../../../投机采样/results/runtime/p0-tf-usable-final-protocol-scope-review-v1-20261016-r1/TERMINAL-INCREMENT.json)。
- [远端实际2070PT核验](../../../投机采样/results/runtime/p0-tf-usable-raw-audit-v1-autodl-i34-20261016-r1/RAW-EPISODE-VERIFICATION.json)；[RNG静态来源审查](../../../投机采样/results/runtime/p0-tf-usable-rng-source-audit-v1-20261016-r1/RNG-SOURCE-AUDIT.json)。
- [TF_CONT训练产物根](../../../投机采样/results/runtime/p0-tf-usable-train-v1-20261016-r1)及[短状态](../../PROGRESS.compact.md)。
