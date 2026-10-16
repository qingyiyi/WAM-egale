# 夜间 draft 与接受路径研究：统一结果

状态：夜间有界实验及结果独审已完成（2026-10-15，Asia/Shanghai），新权重尚未定版；已完成项目目标、原因和下一阶段估时汇总。算子融合已撤回，原tile代码与HW保持原样。本文是研究汇报，不是论文初稿。

## 当前能确定什么

项目目标是：冻结WAM大教师，用小draft提出内部动作去噪候选、批量验证并提交合法连续前缀，通过回退和bonus减少大模型调用；在实际闭环中同时检查任务成功与环境步数，再在相同条件下测完整policy请求速度。当前实现已有正式单chunk收益；全任务质量、稳定新权重和跨模拟器／模型复现仍待补齐。新算子融合已经撤回，本轮不把它算作收益，原tile研究保持原样。

现用 A_off MLP 暂时保留。TF64 是值得扩大训练的结构方向，但本轮已经完成的权重尚未通过多 seed 准入，不能当作已定版的部署模型。结构规格为 16 个位置 token、宽度 64、4 heads、1 层、FFN256、无 dropout、两个对称 proposal heads；53,070 参数，对照 MLP96 为 52,928 参数。参数匹配并不代表 FLOPs 或推理成本相同。

不能用训练 loss、接受率或训练集上的轨迹误差，保证真实任务成功率和环境步数保持。每个 action chunk 仍提交原来的 50 个积分更新，投机减少 teacher forward；环境从初始化到成功执行多少动作，是另一个必须实测的指标。无需人为补动作让长度相等，需报告实际成功、超时及配对长度变化。

## 本轮已经完成的训练结果

历史数据对照九组已完成并独立逐模型复算，保持 current512 暴露与 KD 不变，另加128条辅助 DATA，一次640条 forward，三 seed 各1000 updates。

| 方案 | 代理 root 中位数/630 | 旧通过丢失中位数/527 | 实际 CPU 后继 root 中位数/630 | 通过原门的 seed |
|---|---:|---:|---:|---:|
| TF64 current aux | 525 | 13 | 387 | 0/3 |
| TF64 historical aux | 531 | 9 | 396 | 1/3 |
| MLP96 historical aux | 507 | 24 | 366 | 0/3 |

TF64 historical 在相同 seed 的三组匹配对照中比 MLP96 多23/33/17个代理 root 通过；有结构选择信号，但仅一个 seed 通过，不能挑这个 seed 部署。真实 CPU 后继与代理 Euler 门也不是相同指标。

接着保持所有样本、初始化、抽样和其他 loss 项不变，只把保持旧行为的 KD 系数改为4或16，完成十二组。

| 结构 | λ1 root中位数 | λ4 | λ16 | λ4/16的旧通过丢失中位数 |
|---|---:|---:|---:|---|
| TF64 | 531 | 521 | 510 | 15 / 23 |
| MLP96 | 507 | 516 | 506 | 16 / 23 |

新增十二组全部失败原准入，四配置均不入围。加大 KD 权重没有修复 TF64 的保持问题。九组与十二组的 checkpoint、全部1890条 DEV预测、原 scheduler/project 后继、43分组、原六项门和三 seed 选择均已经独立重算。它们是局部查询指标，不是 episode 成功率。

证据：[历史辅助结果](../../../review/OMISSIONS.md)、[九组独立审计](../../../review/OMISSIONS.md)、[十二组独立审计](../../../review/OMISSIONS.md)。

零起点残差九组也已完成：冻结原OLD CPU模型，只训练零末层修正；TFλ1、TFλ4、MLPλ1各三seed。3906 TRAIN＋1890 DEV初始化真实predict与OLD逐位一致，signed-zero和梯度另有实际检查。训练后独立逐checkpoint复算17,010次真实predict与CPU后继、43分组和全部原准入。已记录实测小计514.904秒；旧814.90字段还含300秒失败prepare保守预留，另30秒补检预留后的保守预算合计844.904，均不能全称实耗。实际runtime总参数包含冻结base，为TF78,998／MLP78,856。

| 残差方案 | root中位数/630 | 旧通过丢失中位数/527 | 实际CPU后继root中位数/630 | 过原门seed |
|---|---:|---:|---:|---:|
| TF λ1 | 516 | 12 | 369 | 0/3 |
| TF λ4 | 522 | 5 | 377 | 0/3 |
| MLP λ1 | 512 | 19 | 366 | 0/3 |

TFλ4的root分别525/522/521，全部低于旧527。旧通过丢失单项达到1%附近和实际后继中位数略增，均不能代替其他准入条件。仍KEEP_A_OFF。证据：[残差结果](../../../review/OMISSIONS.md)、[独立残差审计](../../../review/OMISSIONS.md)。

进一步仅降低TFλ4学习率为1e-5／3e-5各三seed，保持同初始化、draws、损失和1000updates；六组全部失败原准入。两档root中位数均523<OLD527，旧通过丢失中位数4／5；真实CPU后继root为376／379。三seed合计丢失由1e-4的16变为13／12，但seed22反而从4增为5，不声称逐seed都改善。独立重载11,340个DEV查询，stage合计463.02秒<1800。迄今九历史辅助＋十二KD＋九残差＋六学习率共36组完成，暂无稳定的替换配置。证据：[学习率配对结果](../../../review/OMISSIONS.md)。

## 接受标准与路径实验

冻结九份实际日志、2920棵树，两档阈值共5840次比较。相同“最长连续合法前缀，然后最小已接受成本”目标下，DP 与现有穷举逐树完全等价；不能把换成 DP 本身写成新收益。原τ下，greedy 的接受深度数量相同，但改变80次实际已接受 parent；累计预算 DP 将300棵树从接受两步改为一步，DEV平均前缀深度由1.585降为1.488。是否改善闭环，只能由新的完整任务回答。

原 scheduler、scatter/project/compact 和 BF16 后继逐位精确的零残差门，在3906条 TRAIN 查询上为0/3906行、0/7812 head通过。严格精确门目前无法保留加速收益。28个 CPU完整chunk检查通过，覆盖 frame0/4、50更新、bonus、原OLD、路径选择、KV、RNG和PF2周期；CPU共同attention参考不是native CUDA验收。

原生GPU十二段数值检查已通过，4种接受方式各RAW/O3、PF2的spec/spec/target与原T对照。现场比较全部30层有效KV字节；持久化的是KV哈希和元数据，未保存KV张量，不声称能够离线重算KV字节。CPU独立重载六对，960个tensor、12个array、405254个scalar比较全部相等，664个teacher intents闭合。

GPU严格零门未接受近似边，但其完整归一化动作仍与原T有0.00390625最大差。原因范围是批量B14 verifier与原CFG2执行路径的数值差异；本轮不把局部精确后继或τ=0写成相对原策略的无损保证。

证据：[接受路径报告](../../../review/OMISSIONS.md)、[native输出独立重载](../../../review/OMISSIONS.md)。

## 完整任务验证与新训练数据

已冻结六臂T、OLD、GREEDY、BUDGET_DP、PROJECTED_ZERO、PF2_TIGHT，task0/1 × init18/19，每臂至多4个完整任务；双EGL，统一seed12345，完整800 horizon，无截断。先init18两任务；候选相对T与OLD均过原门才继续init19，T与OLD始终完成两个init。失败按800步计，配对成功损失必须0，成功数不降，总惩罚步数比≤1.05、最差pair≤1.10。另报绝对长度变化，原门并不是对称±5%。

24个完整任务及最终独立审计已完成，六臂各4/4成功，没有跳过或重试；五个投机臂相对T与OLD的原四pair门全部通过。每臂只有task0/1 × init18/19四个开发pair，不能声称全LIBERO非劣。

投机臂全部使用冻结的现用A_off权重，比较的是接受器和刷新方式；这24个任务没有验证45次新训练所得TF权重。init17的两条TF研究生成器任务属于TRAIN采集，不能补作配对成功率证据。

| 方案 | 四pair总实际步数 | 相对T总变化 | 相对T最大pair绝对变化 | 全部target调用数 |
|---|---:|---:|---:|---:|
| T | 1107 | 基准 | 基准 | 5246 |
| OLD | 1099 | −0.723% | 3.741% | 3527 |
| GREEDY | 1089 | −1.626% | 3.731% | 3492 |
| BUDGET_DP | 1106 | −0.090% | 3.214% | 3657 |
| PROJECTED_ZERO | 1123 | +1.445% | 3.214% | 5394 |
| PF2_TIGHT | 1110 | +0.271% | 5.714% | 4381 |

GREEDY可作为下一轮冻结的研究候选，保留OLD对照；本轮未自动切换部署preset。其target调用比T少33.435%、比OLD少0.992%，但CFG2／CFG14和实际轨迹不同，不能换算为新速度。ZERO的3431棵树全部depth0、没有接受边，target调用反而增加2.821%；质量过门不等于有加速价值。PF2_TIGHT有22个原T chunk，全部原CFG2/50更新/51action calls实证，未显示这四pair里的质量优势。

调用数拆解还显示GREEDY为71个chunk，OLD为72个；target／chunk分别49.183与48.986，GREEDY反而略高0.402%，action target／chunk也高0.761%。所以这批较少的总调用与少一个实际chunk有关，不足以证明greedy的去噪或单chunk效率更好。证据：[按chunk拆解](../../../review/OMISSIONS.md)。

独立审计重算24episode、659份源／实物artifact，验证12波同UUID、同采样行的双graphics client＋双READY compute；25model、26361intents、6522.86 GPU秒精确闭合，own进程清理完成。初始／最终MuJoCo NPZ重算字节；观测／Python和NumPy RNG只有持久化hash，GPU KV仅保存hash和现场比较记录，不能冒充全部实物离线重算。证据：[最终闭环独审](../../../review/OMISSIONS.md)。

init17双任务新TRAIN数据的原night准备版在入场时被外来GPU占用，以0模型、0GPU秒、0intents失败并保留。质量阶段后续接job使用真实新carry，CPU-only seal及20项metadata检查通过；但03:06两个server在冷构造时均因candidate_predictor提前访问reset才建立的action_per_frame字段失败，0完整episode、0标签。新实耗33.67 GPU秒；两worker缺闭合counter，按原约定保守收费8000intents，累计37774.53 GPU秒／181879intents，own清理通过。旧四段OFF/ON证明是在已reset的snapshot上安装，未覆盖此冷入口；不能拿它掩盖采样尚未发生。

冷启动修复已经在独立隔离源中完成，原25source、失败job和收费保留。原构造／public reset／首policy CPU真实路径检查通过，再以新源运行4个新的原生OFF/ON完整chunk，172intents、输出／路由／RNG／现场30层有效KV相等。04:00:39两条完整EGL TRAIN任务均成功：task0/1、init17为286／262动作步，19／17chunk。实际采到348＋309＝657棵树、1971条同查询teacher标签，全部保留，没有额外teacher前向，也没有把oracle init3/4放进TRAIN。

恢复阶段3model、1761intents、591.249912478 GPU秒，自有GPU/EGL进程清理通过；该冷恢复阶段闭合时累计38365.78347003646秒／183640intents，旧失败8000未退费。独立4.037秒全量重载657文件／1971行、36chunk、全部query/node/parent/步长/marker；另外两对原生输出320tensor／4array／1628scalar逐位相等，154 NVML样本中的67条同UUID双compute＋双graphics证明通过。后补/proc身份审计确认所有原GPU/EGL身份消失，7个根进程已reap。server launcher在FINISH_READY后由SIGTERM清理，不能把这种退出写成自然exit0。证据：[新数据完整独审](../../../review/OMISSIONS.md)。

这两条是未准入TF64研究生成器产生的TRAIN轨迹，没有配对原T质量对照，不能据此宣布TF64成功率或长度不变。随后新aux九组已经全部完成1000updates，并独立重跑17,010条真实predict＋原CPU投影、43×2分组、六门、多seed选择和rank；结果全部吻合。实际prepare6.707＋计划独审7.370＋外层训练/导入/清理455.537＋最终独审47.304＝516.919秒，总stage低于1800。9组均无partial／retry／GPU训练／部署。

| 新数据匹配组 | proxy root中位数/630 | 旧proxy通过丢失中位数/527 | 原CPU后继root中位数/630 | 原CPU旧通过丢失中位数/374 | 完整准入seed |
|---|---:|---:|---:|---:|---:|
| TF current辅助控制 | 525 | 13 | 387 | 31 | 0/3 |
| TF 新查询辅助 | 528 | 12 | 390 | 32 | 0/3 |
| MLP96 新查询辅助 | 507 | 22 | 372 | 30 | 0/3 |

TF新查询组相对current控制的三seed proxy root差为＋3／−2／−1，旧丢失差−1／＋1／＋2；不是一致改善。其root中位数比OLD527高1、mean L2中位数改善，但task0中位数261低于OLD265，旧通过丢失12/527＝2.277%高于原1%门。原CPU后继虽多16个root通过，也丢失32/374＝8.56%的旧通过，同样只能作为局部诊断。三个TF current控制的模型参数与历史对应控制逐字节相同，不能把重复控制当额外独立seed。本轮合计45次CPU训练执行（14个非重复配置、42个配置×seed单元，仅三个seed策略），所有配置尚无稳定替换资格；无新candidate pilot。证据：[新数据九组完整结果](../../../review/OMISSIONS.md)、[最终独立复算](../../../review/OMISSIONS.md)。

另有只读数据诊断：3906旧TRAIN＋1890旧DEV＋1971新TRAIN共7767个实际227维FP32输入，逐字节均唯一，新TRAIN与旧TRAIN／DEV精确重合均0。没有找到同input异teacher或两head覆盖不可能的实物反例，不能据此认定本轮失败来自输入信息不足。该检查不用于训练或阈值选型，也不把compact input哈希当完整WAM query身份。证据：[输入冲突诊断](../../../review/OMISSIONS.md)。

残差九组及学习率六组已在上节给出实际失败结果。CPU旧模型一致不等于GPU原预测器一致，未来GPU适配仍需单独前检和闭环。

同原τ的PF2／PF3／PF4刷新周期对照，真实CPU全周期、原public infer/cache接口及手工原投机／原T对照已经通过；固定native输入只有一个请求和snapshot，不能声称已有连续GPU周期。revision3及launcher R3-R1的异常边界已修复、验证并独审，原数学源不改。新训练无eligible后正式选择该互斥分支，计划为24个同起点数值分支对照＋6个完整EGL任务；后者复用task0/1、init18，每臂仅2个开发pair，不是新holdout，GPU≤3000秒／26000intents。

04:37:43实际GPU入场后，旧O3 prefix_kernel.run从原experiment模块读取QUALITY授权flag，而新launcher只设置REFRESH flag。在1个手工原分支完整chunk完成后，调度分支首次调用原kernel时被defaultOFF拒绝。CPU检查曾只load_native，没有实际走到kernel.run guard，因此未发现这条集成缺口。04:38:44任务FAILED：1个手工完整chunk、0完整数值pair、0episode；手工45次target调用，调度分支23次尝试中22次返回、1次报错，合计68次尝试／67次返回。实耗60.875895秒／1model，失败worker保守收费2000intents，自有清理通过，root67338已exit1/reap。失败、旧11源／R3／plan／AUTH不改，不能把尚未发生的周期结果标通过。该计数已独审，并以004-CORRECTION-001追加更正，未改原冻结004。

隔离授权binding修复通过真实原kernel guard的CPU正负例和独立审查：验证新job的精确plan／AUTH、worker身份及预留后，才在该worker发布原QUALITY／REFRESH标志；旧开关单独不能授权新job。原kernel、binary和R3数学不变，native组件handoff只改phase元数据，实际新plan不改。正式seal27.967秒、独审15.553秒，root再执行完整CPU contract复核。

新阶段05:02:21入场，05:33:46完整闭合，1884.534562355秒／7131intents／7models。24段原生完整chunk、12对手工／新调度输出通过；随后同原τ、同A_off的PF2／PF3／PF4各在task0/1、init18跑两条完整EGL，无跳过／重试／截断，六条全部成功。表中T／OLD／PF2_TIGHT是已核验且初始元数据匹配的旧参考；PF2_TIGHT同时改τ，单独标记，三条新PF只改周期。

| 方案 | task0／task1环境步数 | 相对T合计变化 | target调用合计 | 相对OLD调用变化 |
|---|---:|---:|---:|---:|
| 原T | 280／268 | 基准 | 2586 | +42.95% |
| 连续投机OLD | 288／269 | +1.642% | 1809 | 基准 |
| PF2_TIGHT（周期＋收紧τ） | 296／264 | +2.190% | 2189 | +21.01% |
| PF2（原τ） | 284／271 | +1.277% | 2051 | +13.38% |
| PF3（原τ） | 284／269 | +0.912% | 1945 | +7.52% |
| PF4（原τ） | 282／272 | +1.095% | 1893 | +4.64% |

三条新PF均通过相对T与OLD的原门；相对T的最大单pair绝对变化不超过1.493%。每臂仅两个复用开发pair，没有成功率显著性或总体非劣结论。PF4比T少26.80%调用，但比OLD多4.64%；CFG2／CFG14与实际chunk数不同，不能将调用数换成速度。当前没有证明定期刷新优于连续投机，部署继续保留OLD＋bonus；也没有实测“按接触／动作类型决定开启投机”的新选择器。

独审实际重载24份输出（1,163,110,512字节），1920个tensor对、24个array对、797088个scalar对逐字节相等，保存的RNG张量也实读。现场30层有效KV值比较及持久化hash相符，未保存完整KV值，不能声称离线重读它们。6条新episode＋6条参考的12份commands、24份初末state NPZ实物、周期／50更新／原T的51action调用均核对；108policy chunk＝82spec＋26T、5400更新、1849树。416个NVML样本中各波70／67／66条同UUID双compute＋双graphics通过，全部原进程身份消失。六个server在FINISH_READY及CLOSED后收到SIGTERM，launcher exit1按原样保留，其余根exit0；不能把所有server写成自然退出0。独审28.009秒。证据：[周期完整独审](../../../review/OMISSIONS.md)、[配对原始计数](../../../review/OMISSIONS.md)。

## 论文与扩展范围

[FlowCast，ICLR2026正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html) 为同查询velocity误差与连续前缀提供直接先例，其图像/视频结果不能保证机器人任务成功与环境步数。我们的步长加权L2与MSE在相同d维坐标上可换算，但不能直接复制别人的τ、σ量纲或随机DDPM的严格分布保证。

原τ=0.007839680526450805来自2026-09-29的task8、beam1两层successor-L2合并q90，task9仅做局部统计；它是历史误差分布的校准值，不是90%成功保证。夜间保持原值以隔离算法差异。正式论文需另设不参与最终测试的校准集，用预声明的小网格和闭环成功／惩罚步数选择阈值，再冻结测试；当前不能称已有通用安全阈值。精确来源及数学边界见[接受标准溯源](DRAFT-DESIGN-ACCEPTANCE-LITERATURE-20261013.md)。

优先LIBERO-Long全10任务与RoboTwin两个模拟器，再补Fast-WAM第二种多步action expert结构。当前Long权重不能外推LIBERO100或四套40任务；RoboTwin有[官方LingBot-VA权重](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)，RoboDojo只核到适配和后训练流程，未核到直接匹配的官方VA权重。Fast-WAM官方动作expert仍多步，Flash-WAM公开1action-step学生的跨去噪步投机空间小。真机是否加入取决于是否主张真实机器人部署/实时控制；CVPR没有统一实机硬条件。

[CVPR2027官方CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) 已确认全文11月16日AoE、注册11月10日AoE。内部安排北京时间11月1日18:00给老师完整稿；10/17定算法候选，10/21定独立质量结论，10/26决定用户后续授权的verifier优化是否纳入，10/30冻结主要图表。现在可以逐段写中文方法，按用户要求不生成论文初稿文件。算子侧本轮继续暂停。

Flash-LIBERO是GPU物理后端候选，保留任务与公开入口；它与Flash-WAM是不同项目。当前snapshot客户端读取sim.data._data，作者GPU facade明确不提供该CPU私有handle，因此不能原样替换。已有130任务定义／部分init的静态字节比对，不等于动力学、图像或闭环成功率等价；目前继续用原LIBERO。见[已有源码审查](../../../review/OMISSIONS.md)及[作者仓库](https://github.com/Catigeart/Flash-LIBERO)，本轮没有安装或运行它。

详细官方权重、基准和正式论文证据：[文献核查](../../../review/OMISSIONS.md)。

关于训练集与模拟评测的关系：[WAM训练与闭环直接依据](../../../review/OMISSIONS.md)列出LingBot-VA与TS-DP原论文训练／评测流程。本项目的大教师保持冻结，不能把draft离线训练替代机器人闭环验证。

## 新权重尚未定版的原因与最少补项

已经实际发生的阻断是多seed旧通过保持及分task root门失败。45次执行对应14个非重复配置、42个配置×seed单元，共用三个seed；全部配置未达到至少2/3 seed及中位checkpoint自身准入。最常失败的是task0门（42个单元中41个），不是已证明的模型信息冲突；原threshold、门和部署权重保持不变。逐seed六门、真实后继诊断及计时口径见[完整训练选型表](../../../review/OMISSIONS.md)。

离线未准入不等于已证明新模型的真实任务成功率降低；它表示本轮没有满足预声明的替换条件，也没有获准进入新权重的独立闭环pilot。不能拿现用A_off的质量结果替代这个缺项。

只读溯源还明确：A_off继承父模型task0–7/init0的6321行训练、task8/init0选模历史；本身续训仅task0/init0的833行。TF研究生成器虽从零初始化，仍使用A_off的TRAIN-only蒸馏，且历史辅助池已包含原task9/init0测试标签1029行，须重分类为TRAIN。150份冻结历史episode文件中，标准task0–9 × init0–49全部500组合均有completed记录；这只是历史执行暴露，不说明全部用于梯度或选型，也不是本轮500个质量结果。不能直接宣布init20–49全项目未用。下一阶段先追踪用途并冻结协议；标准全50初态评测仍可做，但如实披露历史接触，独立测试声明另需使用／决策表、初态文件身份、旧teacher身份及WAM训练暴露证据。见[数据划分溯源](../../../review/OMISSIONS.md)。

最少下一步是先登记完整历史split，在当前候选同查询TRAIN尚缺的Long task2–9各增加至少一条完整TRAIN轨迹；保留全部失败／超时及同查询标签。若原3906基础TRAIN、归一化、current512抽样、KD、初始化与数学源码完全不变，可复用已逐字节相同的三条TF current控制，只新增TF64／MLP96扩大aux各三个seed，共6个训练单元；若改变基础协议，必须重新做匹配控制。固定旧采集入口只接受task0/1、init17、两episode，下一阶段要新隔离实现与数据合同，不能直接重启旧job或沿用已关闭AUTH。

按今晚已测周转，8条新增TRAIN约36分钟、20–50条约1.5–3.8小时，另需CPU准备／独审／原生适配；九组训练加审计实测约9分钟，扩大DEV后时间需重估。入围后独立50个pair（教师与候选共100episode）约7.55小时。以上是六臂混合、冷模型／reset／数值前检摊销外推，不是两臂已测时长，不保证下一轮一定通过。若仍不入围，再根据逐task／step失败设计独立目标函数实验；无法给稳定权重完成时间的确定承诺。

原A_off完整chunk除reset的短/中/长KV加速1.602×/1.544×/1.500×仍是已有正式结果；本轮双worker质量wall和数值前检不形成新的性能加速声明。

本批24质量episode的诊断reset均值235–243秒，原T／OLD／GREEDY分别占其model请求wall的53.8%／60.7%／61.3%；新TRAIN两条为60.0%。这是同CPU1–7配额的双worker周转口径，不能与之前单独记录的约125秒诊断直接当作回退或优化比较，也不证明reset耗时完全由CPU或GPU造成。更快GPU未必按算力倍数缩短全量采样；实际无reset policy请求、包含reset的总任务wall仍分别报告。证据：[26条实际reset记录复算](../../../review/OMISSIONS.md)。

下一阶段结构精确规格、split／采样／训练／独立质量与计时条件，以及按实际周转外推Long10×50×两臂约75.5小时的局限，见[扩大实验准备说明](NEXT-SCALE-READINESS-20261015.md)。本轮新增30条质量episode仍只覆盖四个开发task/init pair；两条新TRAIN另列，不能合并成32个独立质量样本。


## 夜间收口和下一阶段顺序

本轮GPU分配阶段累计新增9093.194296秒（约2.53小时）／45253 charged intents；最新全历史累计40311.193927秒／192771 intents，夜间38个model launch。旧冷构造失败8000、旧PF授权失败2000收费完整保留，没有以成功恢复退费。protocol elapsed包含加载／reset／守卫和清理，并非CUDA kernel时间。[最终账本与自有进程检查](../../../review/OMISSIONS.md)重算全部carry链，142个已固定CPU／GPU／EGL进程身份均消失；无新GPU队列，未向外来进程发信号。

下一步按“数据用途登记→补8个任务同查询TRAIN→6个必要的新训练单元→入围后的独立完整任务→正式新速度”推进。已实测结构方向是TF64，现用权重仍A_off；两个概念分别记录在[结构合同](../../../review/OMISSIONS.md)。可以扩大训练采样，不能将未准入TF直接作为论文最终算法展开全量成功率声明。

论文中文方法可以现在逐段讨论，目录可按摘要、引言、相关工作、方法、实验、结论及附录组织。本项目应先讲清WAM同查询验证、合法前缀、回退／bonus与缓存语义，再写已完成的消融和局限；小Transformer＋树投机本身已有TS-DP／FlowCast近邻，不宜单独称创新。方法文稿依用户要求留在对话，不生成本地初稿。

到11月1日老师稿约17天。建议10月17日前完成补数据和匹配训练、10月21日前收口第一轮独立质量，随后先完成Long全10及RoboTwin局部／Fast-WAM适配，再决定扩大范围。当前仅两任务周转已测，RoboTwin、Fast-WAM及自设计MegaTransformer的完成时间尚无实测依据；两模拟器、第二模型和算子全量同时完成存在进度风险，应将算子作为后续独立贡献，避免老师稿依赖未完成收益。算子侧仍按用户要求暂停，原tile材料保持原样。


HW正式收口（2026-10-15T05:50:46+08:00）：已追加[一个研究记录](../../../review/OMISSIONS.md)并同步14份导航增量，未改旧记录。最终核对13个受保护项、47个原tile材料、218个历史记录以及14份导航的原字节均保持，7个撤回活动目录／42文件归档／11标记符合清单；219份record经官方loader校验，新增YAML与本地链接校验通过。见[保全审计](../../../review/OMISSIONS.md)与[格式及链接审计](../../../review/OMISSIONS.md)。
