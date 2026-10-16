# WAM研究单文件阅读包

快照：2026-10-16下午；当前TF retention=false。这是选定原报告的阅读副本；原始数据未附，不能把本文当成新的实验或证书。源码阅读顺序见仓库README，完整评审任务见WEB_REVIEW_PROMPT.md。

---

## 当前评审入口

来源：`review/START_HERE.md`。

# 给外部研究评审者的入口

评审对象：冻结LingBot-VA/WAM上的连续动作投机，以及多候选verifier的Transformer调度。目标是CVPR，内部中文稿节点2026-11-01 18:00（北京时间）。

## 先确定你真的读到了快照

先报告仓库提交或`SOURCE_MANIFEST.json`中的snapshot_id，以及实际打开的文件；不能读取GitHub时直说，不得从标题编造已读源码。用户可改用`REVIEW_PACK.md`或下载当前commit的ZIP。

事实优先级：最新原RESULT/事前协议→对应独立审查→最新研究报告/短状态→旧报告与主题记忆。较新记录不能追溯更改旧FAILED；旧时间点的“正在运行/扩大候选”不代表当前状态。

## 当前事实

- 本机50完整horizon800已经结束、资源清理完成。fresh三臂各10/10，TF长度三项通过；风险task4/init26出现Target成功而TF失败，原retention=false。
- 28个质量配对保存物理/controller/实际初始Torch RNG一致，初始相机25/28。否决task4这一对本身双相机也exact。未保存模拟器内部与后续完整随机状态UNKNOWN。
- 当前无GPU作业、无CPU waiter；AutoDL已归还，不再连接，实例关机未知。明天120局入口SOURCE_READY但实际前置拒绝，未注册、未运行。
- TF结构可暂冻用于算子研究，权重/训练配方未accepted。MLP只在本轮fresh有公平对照，未在本轮8个风险组取得匹配结果。没有最终架构优越性结论。
- 全局旧A_off不变；engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false，新TF正式45chunk受阻。

## 三条不能混淆的证据

1. 本轮新TF质量、晨间80、旧child50是不同权重/范围；不得混池替代独立TEST。
2. 同7个冻结query的逐个验证/B14批量对照证明批处理经济性，不是原始顺序机器人闭环加速。
3. 新attention原型在强B14上的有限改善，不等于cached完整chunk、TF闭环或E2E获益；普通Graph/shared-prefix/共同缓存不是新增贡献。

## 待解决的最小缺口

当前task4已有20个初始数组及首保存观测相同，首环境command索引0即有差；失败与另一个成功初态的接受深度/σ计数相近。缺已有teacher root/候选velocity到提交动作、环境命令的无歧义映射，不能据此决定补训练、累计预算或PF。先前瞻登记2–4对完整诊断；详细日志应有明确20chunk/256query等上限，整个episode仍完整完成。

## 优先阅读

- [当前数字](RESULTS.json)
- [最新50完整报告](../.pipeline/research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)
- [原统计协议](../.pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md)
- [系统与历史负结果](../.pipeline/research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)
- [本轮原RESULT](../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json)
- [本轮独立raw审查](../投机采样/results/runtime/p0-draft-freeze-full50-independent-raw-audit-v1-20261016-r1/SHORTREPORT.md)
- [实际task4诊断](../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/REPORT.md)

其后按问题读取README中的源码，不要求一次读取全部历史文件。未发布的原始大张量仍在服务器，不能声称已在GitHub上重算。


---

## 本轮50完整结果

来源：`.pipeline/research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md`。

# 10/16 draft定型有限验证：完整50局，保留条件未通过

更新时间：2026-10-16 15:16，北京时间。研究报告，不是论文初稿。最新实际证据覆盖晨间的扩大验证建议；原记录、FAILED/PARTIAL、预登记协议均保留。

**结论：TF64结构及7状态/B14 verifier接口可以暂冻供算子研究；seed23的3000更新权重与本轮训练配方没有通过预登记保留条件，不能作为已验收的最终draft。** 新初态表现支持继续研究TF，不能据此宣布优于MLP或已经非劣。明天120局入口已经准备，但真实前置门拒绝；未创建注册、未启动CPU waiter或GPU分片。

## 本轮实际完成

本机A100-SXM4-40GB，单GPUworker、server2/client2；50条完整horizon800：fresh init35三臂各10、已知风险8组T/TF各8、target重复4。实际12:46:40启动，约15:00:57完成资源收尾，记账8056.511秒（2.238小时，含setup/等待/分析，非CUDA活跃时间或正式性能）。总产物2,206,518,844B、计数61859 target forwards，单局最大3700；均低于4小时/3GiB/100000及4000单局帽。58个已观察自有进程身份均已退出。AutoDL已归还，不再SSH；实例关机未核实。

| fresh DEV，10个任务同init35 | 成功 | 实际步数均值／失败800均值 | target前向计数/完整policy chunk |
|---|---:|---:|---:|
| target T | 10/10 | 257.5 / 257.5 | 73.881 |
| TF_CONT | 10/10 | 265.0 / 265.0 | 43.610 |
| MLP_CONT | 10/10 | 267.7 / 267.7 | 43.661 |

TF/T共同成功10对步数比P50/P90/max为 **1.021212 / 1.074884 / 1.092593**；MLP/T为1.029562 / 1.080034 / 1.112840。TF三项新初态长度条件全部通过。前向次数较少不等于实测加速：packed调用同时评估7状态，FLOPs、draft/pack/commit与有效提交成本仍要单独计账。本轮没有正式chunk/task/E2E速度结果。

| 已知风险8组 | T | TF_CONT |
|---|---:|---:|
| 成功 | 5/8 | 5/8 |
| 实际步数总和 | 3610 | 3764 |
| 全任务失败800均值 | 452.75 | 472.00 |

风险配对四格(T,TF)：00=2、01=1、10=1、11=4。task0/init28是T失败796步、TF成功281步；**task4/init26是T成功249步、TF失败796步，触发事前risk_no_new_TF_failure=false**。净成功数相同不能抵消这项预登记条件。其余六项数值检查为true，整体numeric_screen_observed=false、retained=false。风险共同成功4对长度比P50/P90/max为1.090368/1.274578/1.322449，仅描述，不事后追加为新否决规则。失败的实际字段796与固定罚分800分别保存；episode完整性已审核。

## 配对、重复性和原始证据

独立全量CPU审查实际读取3184个PT：1076 actions、1076 latents、1032 obs，34776个数组；全部有限，50份完整commands、成功/步数/统计与原RESULT/PAIRING一致。173来源、两小权重、预登记与491原保护记录未变，三臂及总清理、独立CLOCK均通过。

28个算法比较对的已保存初始sim/controller、RESET与实际target初始Torch RNG一致；相机严格25/28。差异均为task8腕相机：fresh TF5pixel max1、MLP6pixel max1；risk TF10pixel max2。**否决的task4/init26两相机及保存物理/RNG本身全部exact**。这些证明只覆盖实际保存字段；未保存仿真内部状态及后续完整随机状态仍UNKNOWN。

另列target重复4局，不混成独立质量样本：task2/init33成功258/263步，task8/init33成功536/382步；命令均不同，初始腕相机分别1/2pixel微差。相机差异与行为差异共现，不构成根因证明。旧晨间80、旧child50、旧task7/init24失败不混池；本轮没有重验task7/init24。

TF/MLP新权重各在全部20237条真实CAL完成GPU同设备adapter与原decode逐位核验，且CPU/CUDA RNG不变；这不替代闭环验收。

## 固定研究接口与训练事实

[结构接口](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/OPERATOR-INTERFACE.STRUCTURE-FIXED.json)明确只作算子研究载体，权重/配方未accepted。

TF53070参数：16token×17特征，state7、previous-direction7、sigma、next-sigma、t/1000；根使用已有方向，递归使用父draft方向，不能统称previous target。单层pre-norm、D64/H4/FF256、GELU、dropout0、学习位置、两组7维残差头；没有图像/prompt cross-attn，整个WAM仍有视觉条件。MLP52928参数作等容量对照。小模型参数与内部特征FP32，adapter返回current_state.dtype，本轮为BF16；接受打分FP32。

固定seed20260923/final3000，TF权重SHA3657388f…d64ac，匹配MLP为3dc01a48…103b0。训练不是连续3000更新：

1. FIT174 target159691 query，2000 updates、batch512、AdamW lr1e-4/wd.01；任务→episode→行均衡抽样，FIT-only固定normalization。
2. 500 updates，target256＋历史candidate256、lr3e-5/wd.01，重置AdamW，不改norm。
3. 同一起点2500再固定500，重置AdamW，重复原500抽样表，数据/比例/lr/norm不变。TF CPU125.154秒；匹配MLP CPU54.622秒。

历史candidate14997行=3906 A_off＋11091旧TF，不能称当前TF的on-policy数据。损失、来源及完整SHA见接口；没有根据这次DEV挑seed/checkpoint。此次结果未证明增加训练步数是修复方法。

verifier保持K2/B2 bonus、7状态×CFG2=B14、Q16/状态、prefix_then_cost、共享历史/O3、旧sigma加权L2阈值0.007839680526450805；PF关闭。reset仍共同使用512-token BF16 bank。本轮没有集成新算子或文本投影缓存。

## task4有限诊断：根因仍未知，D01 HOLD

[最终诊断](../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/REPORT.md)核20个初始sim/controller数组、target RNG及双相机exact。环境命令索引0已有1/7通道差异、max.0078125；保存actions_0有62个BF16元素不同、max.015625。第一份obs_data_0最早相机差在保存帧3；保存帧不是已证明的同编号环境步，不能直接判某条边是原因。

失败局前14完整chunk、252条记录，提交深度0/1/2为35/11/206；同task成功init35为33/13/206，sigma分箱相同。记录的changed_route=0仅描述已有选择器日志，不是重新验证packed/sequential数值等价。成功参照初态不同；相似计数不能证明安全，也未给出“接受太多/某sigma/训练分布”根因。

现有日志缺逐查询实际teacher root/候选velocity及到最终环境命令的无歧义映射，不能补算未保存数据。因此没有足够证据触发D01、累计误差预算、PF或cross-attn；这些继续HOLD。

下一作业应先在独立入口前瞻登记2–4对完整诊断episode，直接保存已经计算的teacher输出、sigma/边/提交索引及到环境命令的映射，限制最多20完整chunk或256查询的详细日志，不增加teacher调用、不存逐query30层KV。按本轮T132.93秒、TF270.28秒的实际病例墙钟，2–4对纯episode约13–27分钟，运行可留20–40分钟；入口实现/审查另估0.5–1.5小时。新增紧凑日志容量尚未实测，需设明确cap。此作业尚未登记/启动，不是smoke，也不能冒充独立TEST。

## 系统线与放行关系

H1有真实M32/96/224及attention Q16账本，原始瓶颈机制尚需补强；H2同7冻结query的批处理经济性成立，不能称机器人加速。H3现有路线B有强B14上的有限增量：未缓存两进程全档下降4.98%；共同文本投影缓存单进程pilot下降10.27%，尚缺独立复测、生命周期与cached完整chunk。H4当前TF未收口。本轮没有新增这些系统测量；原始证据见[夜间结果](../.pipeline/research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)。

明天算子侧可沿独立命名空间、固定7状态B14接口继续：先复测最佳共同缓存基线并验证动态生命周期，再做真实短/中/长KV的block/30层/完整chunk状态与有效提交成本；原tile与撤回融合保持。优先已有路线B，不启动30层大融合。最终候选权重/引擎/独立TEST仍各自冻结，不借用结构接口作质量证书。

明天init36–39的120局扩大DEV为4×30完整任务；源码/181来源绑定/clock/有界dispatcher/控制器均已审查。真实_today_gate已只读验证拒绝，当前无REGISTRATION/CPU waiter/GPU分片。下一候选需要新的前瞻版本，不能修改旧retained=false放行。TEST40–49未运行/未读结果，留待候选与引擎冻结。用户偏好持续：先质量验证，需要算子正式计时时通知在完整30局分片末暂停；正式计时需无争用。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission四项均false；全局旧A_off不变，TF正式45chunk入口仍受阻。

## 关键路径与止损

默认主线：补齐风险病例证据→仅证据触发一次新FIT/CAL与等预算训练→有限DEV→候选/引擎冻结→独立TEST→质量获准后的正式性能。TF结构暂冻，架构/阈值/PF不继续网格搜索。

乐观：下一诊断能定位可修字段，10/18前恢复质量收口；常规：一次有限数据/训练至10/20–21，与系统复测并行；不利：10/22仍不收口就停止新增研究臂，保留负结果并收缩完整机器人加速主张。三者是排程情景，不是已跑作业或保证。10/23冻结、10/26测试、10/27获准计时、10/28主表均依赖前门；11/01 18:00完整中文稿交老师不变。第二环境/真机不展开。

10/16重新核[CVPR官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)及[CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)：论文注册11/10、正文11/16、补充11/23 AoE，分别北京时间11/11、11/17、11/24 19:59。两页未另列abstract截止，不推断注册字段；所有作者需OpenReview profile，无机构邮箱的新账号审核可长达两周。10/18前检查profile、10/25处理遗留；本轮未发外部消息或改账号。

## 原始证据导航

- [今天PLAN/预算与动作](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PLAN.md)、[原预登记](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PREREGISTRATION.json)。
- [原50报告](../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/REPORT.md)、[原完整RESULT](../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json)。
- [全50原始独审](../投机采样/results/runtime/p0-draft-freeze-full50-independent-raw-audit-v1-20261016-r1/SHORTREPORT.md)、[来源/预算/真实清理独审](../投机采样/results/runtime/p0-draft-freeze-terminal-independent-review-v1-20261016-r1/REVIEW-RESULT.json)。
- [有限诊断最终JSON](../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/RESULT-FINAL.json)、[未来入口被阻塞状态](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/FUTURE-VALIDATION-READY.json)。
- [晨间80与原PARTIAL](../.pipeline/research/speculative-sampling/TF-USABILITY-RESULT-20261016.md)、[历史系统/质量负结果](../.pipeline/research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)、[原统计协议](../.pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md)。


---

## 独立统计与数据隔离协议

来源：`.pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md`。

# 后续统计质量评估协议

登记时间：2026-10-15 20:55，北京时间。当前fresh30 r2仍运行；登记时已知T10/10、两阶段MLP9/10，未读取任何两阶段TF的完成局结果。本文登记后续统计分析/扩大质量的口径，不修改r2源码、原像素门、原FAILED或当前AutoDL/正式bench准入；需要实现接续时须用明确的新版本，不制作假的r2 RESULT。

## 文献依据与命题

[LIBERO官方评估](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/evaluate.py)固定任务初态/seed/最大步数并统计成功率；[OpenVLA官方评估](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)同样按task/init运行完整局，默认50局/task，done时提前结束。所查脚本没有跨策略首帧相机逐位相等或实际环境步数完全相等的门。其事实只支持统计基准评估的设计；不能推出不同观测下的严格逐局因果或逐值复现。

后续须同时保留两种证据：统计质量报告成功率、实际环境步数和区间；数值复现检查同一真实输入上的输出/路径/全部有效KV及RNG逐位一致。二者各自注明控制条件。原r2像素门按预登记执行，失败仍是失败，不将官方评估设计追溯变为原协议PASS。

## 固定条件与失败优先

1. 同target权重、suite/task、官方init索引、环境/策略seed、相机尺寸、settle配置及原800 horizon；逐局完整执行，无截断/事后挑局/换seed。σ/τ、K2/B2 bonus和prefix_then_cost及训练checkpoint均不改。
2. 实际sim/controller/原始init state数组、环境Python/NumPy RNG、target初始CPU/CUDA RNG及完整reset状态必须逐项相同。以原数组及实际RNG为证，不将含obs hash的复合bool误当纯物理状态布尔。所有来源/原保存输出覆盖/有限值/target计数及终态清理均需通过。
3. 两相机实际uint8数组与各自metadata哈希仍必须吻合；跨策略逐相机报告不同像素/通道数、最大/平均差及区域，不替换图像、不设观察结果后选择的像素容差。非逐位相等的局标明不能作完全相同观察的强配对证明。
4. 若全部30完整、所有自有进程清理、来源与上述物理/RNG控制成立，仅原像素门失败，可另生成明确标为“统计质量/物理初态配对”的独立分析。原FAILED/PAIRING证书原样随附；其他运行、预算、数值、缺文件或RNG/物理状态失败均不得出完整资格证书。独立产物不能命名或伪装为原r2 PASS。

## 固定候选工程门

保留此前同一成功/步数条件：TF成功数不低于同批T及matched两阶段MLP，双方成功非空；TF/T实际步数比P50≤1.10、P90≤1.20、max≤1.30。MLP只在成功数不低于T且同样步数条件满足时作算法对照；登记时已见MLP9/10，因此本轮它不满足成功数条件。所有失败按800固定罚分另报，不能只删失败局或仅报双方成功的均值。

这些阈值是工程筛选，不是文献保证。10个DEV配对不证明3个百分点非劣，原40/新30均为开发数据。通过只推荐固定候选进入新的Long10×init25–29的T50/candidate50独立完整质量；扩大数据须同样报告每task成败、配对胜负、成功局步数分布、全体失败罚分和区间，AutoDL墙钟仍不作正式性能。

正式完整chunk继续要求同一真实输入的RAW/O3输出、全部30层有效KV、接受路径、scheduler/RNG逐位一致，以及本机独占GPU。更改统计质量资格的接入必须是新namespace/新证书、独立审阅；不改原benchmark或用描述性统计冒充数值复现。

## 同seed父子消融与探索性扩大评估登记

补充登记：2026-10-15 21:26，北京时间。此时r2全部30局终态，原相机门FAILED保留；TF23的2000步parent尚未运行或读取任何新闭环结果。仅对这一真实parent增加Long10/init24完整10局，与同seed child2500和r2 T作开发集消融。不以旧TF22代替parent23，不更改原阈值、seed或800 horizon。

后续可在明确独立的research quality namespace中，选择父/子中一个进行新T50/candidate50探索性质量测量；这不要求将开发集负结果改成工程PASS。前提是父10及r2完整终态、真实来源/数值/物理与RNG资格全部成立，像素差如实列出。预先固定选择顺序：成功数更多者优先；相同则全10局的800失败罚分均值更小者优先；再相同选parent。选择只限这两个已经固定的TF23权重，不加seed或调τ。该选择属于DEV上的探索性选择，不能称独立测试或原CAL中位数选择。

部署证书须独立记录`quality_gate_passed`及原工程条件逐项结果；即便该值为false，也可为测量风险而执行已授权的探索性独立质量任务，但不得称质量合格、最终推荐或覆盖原FAILED。工程门未通过时，后续结果仍需报告全部成败/长度/失败罚分及局限；扩大评估不是非劣保证。原v2与正式性能bench不改，其原有正向资格门仍生效。租卡总预算、原100局完成后才接续、单GPU worker、独立审阅和清理限制保持原约束。


---

## 夜间系统机制与历史负结果

来源：`.pipeline/research/speculative-sampling/OVERNIGHT-RESULT-20261016.md`。

# 10月16日夜间执行结果

更新时间：2026-10-16 05:21，北京时间。状态：本机有限机制、完整chunk正确性、缓存共同基线全部完成并清理；AutoDL新104局完成并清理，原始小档已归档。TF质量门未通过。本文件为研究执行报告，旧FAILED和既定协议均保留。

## A. 动态事实

本机实际完成三档KV、1/3/7状态的完整30层机制矩阵，补齐静态CUDA Graph以及同一7个真实候选逐个B2的整体Graph基线。Q02完成真实TF task7前10个完整policy/cache chunk两遍重放。新跨候选attention原型v1数值失败；保留负结果后，最后一次固定几何规则v2通过有限门，完整30层的长KV局部改善在两个独立进程复现。本夜本机已登记GPU作业均已终止，自有GPU进程均清理；全局A_off、原tile、撤回融合和TF正式45chunk质量门未改。

旧AutoDL T/A_off 100局完整但原严格相机配对FAILED。独立描述性结果：T48/50、A_off47/50，target胜2、spec胜1、双方成功46、双方失败1；双方成功长度比P50=1、P90=1.0454、max=1.5939；全任务800失败罚分均值285.50/298.64。50对物理/环境RNG相同，4对腕相机hash不同；旧像素数组及target初始RNG缺失，不能修补旧证书。它不是TF、不是最终盲TEST、不是正式性能。

新AutoDL独立链于北京时间05:04:01签发完整研究终态，104/104局（4条reset对照+T50+TF50）及自有进程清理通过，未复用旧T。原4小时帽从01:19:05开始，05:14停止工作、05:19完成归档清理，早于用户09:00授权上限；实际新链约3小时45分。脚本清理不等于租卡实例已关机，实例未关闭。

|新50对主结果|T|TF64 child2500|
|---|---:|---:|
|成功数|48/50|44/50|
|全部实际环境步数合计|14301|16255|
|失败记800的均值|286.18|325.58|
|实际policy chunks|931|1049|
|目标模型调用/完整chunk（含video/cache）|73.897|43.513|

配对共同成功43、仅T成功5、仅TF成功1、共同失败1，任务等权成功率差为−8个百分点；固定Long10任务内成对init bootstrap 2000次，描述性95%区间[−18,0]个百分点，不能建立3pp非劣。43组共同成功的TF/T长度比P50/P90/max=0.99617/1.05430/1.40；最大来自task6/init25（245→343步），初始相机逐位相同。预定max≤1.30门与成功数门均未过。TF失败为task1/init29、3/26、4/26、5/28、6/27、8/25；T失败0/28、5/28。本次task7双方均5/5，不能把旧init24失败硬编码成task7问题。

50对实际物理/控制器/RESET与目标初始CPU/CUDA RNG逐位相同；原相机47/50逐位，3对腕图分别task2/init25差2pixel、task8/init25和8/26各差1pixel，最大幅度1，不设容差。严格像素47组仅作敏感性子集：T45/47、TF42/47，差−6.383pp；不能替代50对主表或修补旧FAILED。独审已经重读全部100局原始结果、50对实际初始数组和两相机及真实RNG tensor，见[统计独审](../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/REVIEW-RESULT.json)和[全原始覆盖增量](../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/RAW-COVERAGE-INCREMENT.json)。

真实20237条旧CAL的GPU adapter/decode同设备BF16逐位、finite、RNG检查通过；旧查询离线接受率约94.3%–95.5%，并未换来新闭环质量达标。18896棵实际树depth0/1/2=2543/813/15540，平均接受深度1.6878。目标调用从68798降至45645，却增加实际chunk数量并降低SR，不能把调用下降直接报为任务加速。原始cost构造未重算，CAL接口逐位也不是递归质量证明。

时间仅为共享PCIe机器描述：bank428.38秒，4条reset验证阶段622.17秒；T50阶段6704.53秒（平均134.09秒/局，含阶段间开销），TF50阶段5491.40秒（109.83秒/局）。不是本机正式chunk/task/E2E速度。4条reset原/bank物理及相机数组等价，双方质量均用同一bank；不得将旧reset比例沿用到新账本。

## B. H1–H4判断

|假设|现有证据|边界/缺口|
|---|---|---|
|H1 小action存在shape效率限制|真实输出投影M32/96/224；NCU三次实际launch的SM吞吐约24.19%/38.71%/47.65%，局部行吞吐随M改善|单权重暖缓存profile，只支持局部机制；不能外推全部307个Linear或称HBM饱和|
|H2 同组候选批量验证有经济性|同7个真实query，完整30层顺序B2 Graph与B14 shared Graph比较，约2.45/2.91/3.36倍（短/中/长）|跨独立job的静态fixture机制参照，非正式机器人速度；不同batch的BF16结果不一定相同|
|H3 新调度超过最佳已有B14|未缓存强基线下两进程长档下降9.64%、全档4.98%；新增同形状文本缓存共同基线下，单进程完整pilot短/中/长下降3.86%/6.45%/18.40%，全六query10.27%|新共同基线结果尚待独立复测；缓存生命周期与完整chunk性能未验证；新引擎1/3状态、shape交互R和cached G1控制缺失；不能把原有共享prefix思想当新贡献|
|H4 单位有效提交与闭环收益|新TF50目标调用减少，但SR48→44/50、最大成功轨迹比1.4，当前质量条件未满足|新缓存/调度仍无完整chunk及episode联合成本，需实际ΣT_round/Σg；当前TF不能开放正式性能或宣称完整机器人加速成立|

## C. 实际shape和DAG

target隐藏3072、24头×128、FFN14336、30层。112维紧凑action不是Transformer隐藏维。实际action embed输入为[14,16,30]；普通投影输入[14,16,3072]，库已展平M224。n1/n3/n7的M分别32/96/224，CFG未拆成独立模型调用。attention每实例仍为Q16，单纯batch增大没有将其改成一个互相可见的Q112序列。

每query实际307个Linear：30 block各10个，加7个embedding/projection；305个输入BF16、2个time embedding输入FP32。文本投影/cross K/V输入有[14,512,3072]，M7168，不能将所有GEMM都说成M224。按2MNK统计，文本相关占线性运算量约81%，不是81%墙钟。

|算子|实际M/N/K（B14）|
|---|---|
|action embedding|224/3072/30|
|self Q/K/V/out、cross Q/out|224/3072/3072|
|cross K/V|7168/3072/3072|
|FFN up/down|224/14336/3072；224/3072/14336|
|time embedding两层（FP32输入）|224/3072/256；224/3072/3072|

block依赖：norm/modulation→QKV→QK norm/RoPE→合法历史+私有16-token self attention→out/residual→norm/cross attention(text)→residual→norm/modulation→FFN→gated residual。不能跨未完成归约随意流水。CFG两组公共历史不同，候选间私有KV不可互见。

旧shared kernel已去除历史materialization，但一个CTA处理一个candidate/head，没有CTA内跨候选共享；KV只存一份不代表只从HBM读取一次。未缓存原六份完整profile中cross占kernel duration54.4–62.2%，主要KV投影；旧eager prompt cache负结果保留。该profile不代表本轮新静态缓存后的热点，下一次路线选择需更新profile。

## D. 本轮实际测量

以下以first/last两真实query中位数给范围，单位ms；各arm保留6次样本，profile不混入计时。只有静态真实fixture，不含输入更新/Graph重建/生产cache生命周期。

|总KV|同7query顺序B2 Graph|已有B14 shared Graph|整组验证批处理经济性|单状态Graph|
|---|---:|---:|---:|---:|
|288|166.58–166.62|68.09–68.17|约2.45×|23.86–23.87|
|1152|213.76–213.77|73.49–73.63|约2.91×|30.63|
|2160|265.85–265.97|79.01–79.11|约3.36×|38.03–38.07|

必须采用合理优化的单状态baseline。B14 shared/单状态Graph局部成本比约2.855/2.402/2.078；三个真实A_off chunk每树平均合法推进g为2.526/2.333/2.286。K2+bonus每树最多推进3步，因此短档即使忽略draft/pack/accept等开销、每轮均推进3步，按这里的局部成本也只有约1.05倍上限。该比较是固定first/last成本的敏感性分析，不是已经测得的ΣT_round/Σg或chunk负收益，但显示更强target基线会显著收窄投机收益。

上述1.05倍仅是未缓存静态Graph的局部上限；下节同形状缓存改变了双方成本，不能将旧上限沿用到新配置。

完整三chunk各50次scheduler更新；树数19/21/21，树内Σg48/49/48，加起始普通target更新2/1/2才各为50。bonus/fallback只按真实推进计一次。 [既有完整episode日志审计](../投机采样/results/runtime/p0-wam-existing-episode-kv-field-availability-v1-20261016-r1/AVAILABILITY.json)检查TF10局3611树及A_off10局4738树；现有小日志没有逐轮有效KV长度，不能据frame/chunk编号补造真实短中长权重。本表继续三档等权，下一版采集应记录实际validKV与合法提交数。

Q02：32真实树、224候选、256额外target forward；方向32/32树不逐位，最大差0.3359375，但接受路径/深度翻转0/32；原action/video、完整有效KV/RNG复现。只覆盖task7前10chunk的记录重放，未重建机器人环境反事实。局部方向差异存在，当前样本不支持将其写成task7失败根因。

实验故障保留：完整矩阵r1资源中断，r2入场BLOCKED未启动GPU，r3draft-mode入口错误，r4完成；NCU r1包装pin失败、r2实际采得3kernel但CSV单位行解析使原监督器FAILED，独立CPU报告解析真实counter而不改原状态；顺序Graph r2在首次replay前误读capture输出，修复后r3完成。714为顺序Graph保守Python forward intents，物理设备等价forward为672，不能混写。

## E. 原型决定

本夜只实现路线B的最小数值原型，独立于原tile和生产入口。G1为不跨候选的BM64控制，G2/G4为两种登记分组，共享同CFG prefix，私有suffix逐行mask；不做跨CFG合并、不加稠密mask、不改action horizon、不恢复旧融合。

长KV三层：G1/G2均逐位；G4每层不同822/2050/1577元素，max差0.00390625/0.0078125/0.00390625，停止晋级。扩大G2检查：短KV first/last共66对照全部逐位；中KV first的第0层G2不同1152/688128元素，max0.001953125。监督器完成该原native完整query并核冻结输出/有效KV/RNG后停止：实际67候选对照、90 native calls、3完整model forward，36.76GPU作业秒并清理。

v1版本NO-GO，未计时、未集成。中档prefix1136、G2虚拟K1168与原K1152使BN64分块从18增至19是源码事实；尚未实验隔离唯一数值根因。之后仅做一次独立v2修订：当ceil((P+32)/64)=ceil((P+16)/64)时用G2，否则G1。它是固定几何判据，不按task/失败名单回退，也不构成普遍BF16等价证明。三个真实prefix272/1136/2144分别使用G2/G1/G2。

v2实际完成180层probe及180层实际替换，共12个完整30层forward、540次attention；六个实际替换完整输出与冻结原输出逐位、有效KV/输入/slots/RNG不变，58.45GPU作业秒并清理。随后两个独立进程各完整六query×三臂，不挑长档、不挑最好轮：每进程198 Python intents（其中18 capture不执行）、180设备等价forward，全部18次eager准入先过才计时，所有216个测量replay及36个保存输出的CPU字节核验通过。三臂为native BM128、G1 BM64、guarded_v2；native→G1不能全归因于tile尺寸。

|真实KV|两进程合并native→guarded延迟下降|解释|
|---|---:|---|
|288|1.826%|未达到5%投入筛选线|
|1152|2.895%|guarded本身使用G1，无跨候选分组收益|
|2160|9.644%|native→G1下降4.178%，G1→guarded再下降5.705%，长档增量复现|

六query等权Σ中位数下降，r1为4.8997%、r2为5.0593%，合并4.9796%；不能挑r2写全档稳定超过5%。停止第三次重复和分组调参；随后仅做一次实际attention counter归因和一次独立完整chunk正确性验证。输入更新、Graph构建、动态cache生命周期均不在本表中。[Cascade原始说明](https://flashinfer.ai/2024/02/02/cascade-inference.html)已有共享prefix/query复用、prefix/suffix统计合并及A100评测；不能将本原型的复用思想直接写成新贡献。

随后一次真实long2160/layer00 NCU归因完成并独审：native/G1/G2的CTA为336/336/192，寄存器255/168/168，shared memory每block66.560/50.176/50.176KB；L2请求508.791/477.613/320.893MB，DRAM访问70.193/67.076/66.197MB。G1→G2减少32.81% L2请求，DRAM仅降1.31%，achieved occupancy反而从17.86%降到11.15%。它支持跨候选减少重复片上请求和CTA工作，不支持“HBM减半”或“occupancy越高越快”。该NCU每launch实际11个counter passes，暖输入且cache/clock未控制；407.232/296.480/171.296µs是profile耗时，不能混入Graph表或机器人性能。输入/全部有效KV/CFG/slots/RNG/版本保持，profile和离线导出全部自有进程已清理，GPU作业计费11.13秒。

完整chunk S04目前的三次独立登记如下，旧失败没有被重写或拼接成通过：

|登记|实际结果|原因与后续边界|
|---|---|---|
|v1|FAILED；19 forward、42.8064 GPU作业秒、0/6完整、guarded尚未调用；已清理|观察器误把普通target临时槽metadata当成B14严格只读；失败后差异数组缺失，不能补判“仅metadata变化”或有效KV通过|
|v2观察语义修正|超时FAILED；268 forward、785.8769秒、5/6完整、250提交；已清理|短/中两对严格一致；long/native完整，long/guarded仅普通video21次、无B14/提交；全量状态审计耗时估计不足，不是已记录的模型数值差异|
|同数值预算续测|COMPLETE；6/6完整、300提交、294 forward、938.4385秒并清理|三档全量新配对通过，未拼旧通过臂；三job累计581 forward、1767.1217秒、93,178,320B，均在登记预算内|

普通原生路径会写空闲槽、结束撤销mask；v2核验准确的空闲槽id/is_pred变化，同时保持全部原有效KV、有效metadata及mask不变。B14仍检查全池metadata和版本严格只读。v2部分结果实测40个guarded B14×30层及108普通query×30层；21600个id变化全部位于原无效槽。后续完整预算版补齐长档实际配对，全部六臂均各完成50次提交。

完整预算版已经[独立实际结果审查](../投机采样/results/runtime/p0-wam-s04-budget-independent-result-review-v1-20261016-r1/REVIEW-RESULT.json)：276逐query输入/输出及18历史准备，共294次target；122个B14×30层，其中guarded实际1830层；130普通query×30层准确核验scratch。9对保存的actions/latents/obs共4,793,760元素另用CPU逐字节及finite核验；全量有效KV、RNG、scheduler、VAE/server状态一致。范围仍是三个原A_off snapshot的完整policy/cache重放、0新环境episode，不是TF质量或正式性能许可。三job累计上限600 forward/128MiB，GPU预留2028.68秒小于用户S04总3小时。原900秒协议与v1/v2 FAILED保持，详见[观察纠错决定](../投机采样/results/runtime/p0-wam-s04-observer-correction-decision-v2-20261016-r1/DECISION.json)、[同实现预算续测决定](../投机采样/results/runtime/p0-wam-s04-budget-continuation-decision-v1-20261016-r1/DECISION.json)、[资源续测独审](../投机采样/results/runtime/p0-wam-s04-budget-independent-source-review-v1-20261016-r1/REVIEW-RESULT.json)。

static prompt缓存共同基线v2已经COMPLETE并清理，[独立实际结果审查](../投机采样/results/runtime/p0-wam-static-prompt-baseline-v2-independent-result-review-v1-20261016-r1/REPORT.md)通过。六真实query×六臂=36行，所有36次eager准入先于任何Graph；216次六顺序平衡测量、36保存输出逐位/finite通过，410 Python intents/374设备等价forward，输入/全部有效KV/slots/RNG/参数版本及缓存输出不变。424个实际NVML样本未见foreign进程。单作业325.346秒，新增约0.88MiB，原blocked v1与全部旧失败保留。

|KV|cached单状态C1|cached原B14|cached新B14|新调度对共同cached B14延迟下降|
|---|---:|---:|---:|---:|
|288|19.119ms|31.203ms|29.998ms|3.861%|
|1152|25.949ms|35.737ms|33.431ms|6.452%|
|2160|33.451ms|41.422ms|33.800ms|18.402%|

各档是first/last两query各六样本中位数的均值；全六queryΣ中位数下降10.274%。仅一个独立进程，不能称两进程复现。缓存本身让C1下降20.554%/15.880%/12.648%，原B14下降53.481%/50.217%/46.722%；这是普通共同工程收益，不是MegaTransformer新贡献。cached新B14/C1局部成本比1.569/1.288/1.010，不能替代实际ΣT_round/Σg或完整chunk速度。未测cached G1独立控制，18.4%不能全部归因跨候选分组。

缓存分别保存原B2和B14完整61个投影输出，禁止从B14切片冒充原B2数值。payload合计2.859375GiB，另有64MiB prompt副本；实际allocated peak17.707GiB。首填wall分别B2为92.364ms、B14为697.575ms，包含完整第一次原forward、clone与库first-use，并非独立cache构建成本。旧reset bank保存512-token编码，本实验缓存的是Transformer内部投影；两者须分开计账。当前Graph固定输入已准入，实际输入更新、Graph构建、prompt/权重/CFG变化与cache失效均不含在表中；未接生产，未完成cached完整chunk/机器人质量。旧eager缓存慢17.33%/23.79%/6.48%记录继续有效，不用新静态条件覆盖。

## F. 质量收口与放行

TF64结构可暂冻，child权重只是研究候选。原fresh30的task8腕图像素配对FAILED保持；独立描述性诊断为T10/10、TF/MLP均9/10，TF成功子集长度条件通过但成功数门失败。新AutoDL50对扩大测量也观察到成功数下降与max长度门失败，TF权重/接受器不能作为全量生产版本冻结；TF64结构和7状态接口可暂冻供系统线继续。

96个真实teacher查询与target-only历史CAL同σ对照时，root通过率37.5%对37.24%，第一层反事实候选37.5%对57.78%；较大差异集中未提交节点。进一步查原树，40个不通过的first查询对应父边40/40已经超过τ；在固定父边下仅改善这些子预测不能增加合法提交深度，不代表联合重训绝无收益。

之后补齐[更同类的旧candidate审计](../投机采样/results/runtime/p0-wam-old-candidate-support-cpu-v1-20261016-r2/REPORT.txt)：14997行实际上保留4999完整三节点树、teacher/previous/selected/depth/commit。历史TF aux与当前风险96行按精确σ链及root/first层级匹配，root旧/当前通过率均37.5%、误差均值0.01393/0.01509；first通过率也均37.5%、误差0.02266/0.02146。加入已记录reach类别后仍能匹配；不同generator的head编号并非稳定语义，严格node编号+reach匹配的0分母单独保留，不伪造同节点等价。之前target-only对照的明显劣化在同类candidate对照中没有复现。旧FIT与当前风险DEV仍存在选择/初态混杂，这只削弱“新数据缺口”论证，不证明没有分布问题。

完整当前10局3611树的depth0/1/2为492/155/2964；若只修子边而固定父预测，每树额外合法推进上限155/3611=0.0429。相同风险96行CPU诊断中常速度v_prev局部误差均值0.01830、通过40/96，TF best-of-two为0.01934、36/96；不是GPU接受或闭环效果证明。[FlowCast（ICLR2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)已采用常速度外推和MSE接受，因而必须保留常速度基线；当前96行诊断不等于其完整算法复现，也不能把生成轨迹误差界外推为机器人成功保证。上述两项官方来源于10/16再次核对。D01/D02继续HOLD，不为Transformer外观启动全量训练。扩大质量和原型完整chunk正确性已收齐；下一步核查严格配对的失败/最长病例，并补强target缓存生命周期及单位有效提交成本；若以后有明确可达父边监督缺口，再按一次30完整episode与等预算三训练条件推进。

新50局的[实际提交路径审计](../投机采样/results/runtime/p0-tf-heldout-path-cost-descriptive-v1-20261016-r1/REPORT.md)完成：1049 chunks、18896树、31893条实际提交draft边，113376个候选FP32记录的选路与提交守恒全部通过。只累计选中的合法路径，不加拒绝分支、不重复计算bonus/fallback；没有原velocity，不声称重算score。先逐episode再组平均，44成功/6失败的每chunk累计score为0.0643402/0.0633901，每边为0.00211407/0.00208932；六个有成败的task内，失败局每chunk均值也全部较低。失败episode总和较大同时伴随50chunks，不能由此认定累计误差根因，当前结果不支持直接选累计预算阈值。预登记σ段实际提交边在≤0.2、(0.2,0.4]、(0.4,0.6]分别27939/3244/710，>0.6为0；高σ仍可执行target。此审计不触发新训练或新接受器，也不改变质量FAILED。

实际3,840,000训练抽样索引、FIT/CAL/aux及当前远端来源未见init30–49；benchmark500只有prepared/preflight。可以预留本项目FIT30/31、CAL32、DEV33/34、TEST40–49；上游target预训练完整来源未知，不能声称绝对未见。25–29扩大结果解盲后不能改称最终盲TEST。

当前派生状态为`engineering_quality_screen=false`、`independent_test_complete=false`、`noninferiority_established=false`、`formal_performance_permission=false`；其中独立TEST指未来冻结后的主TEST，不能把已解盲的25–29研究测量改名。冻结候选/接受器/树/引擎和TEST清单→独立TEST→按原协议质量放行→本机独占正式性能。工程筛选、独立TEST完整性、3pp非劣结论、正式性能许可四项分开；50对即使零新增失败，也不能仅凭非显著宣称3pp非劣。

## G. 默认关键路径

10/16已收齐扩大质量及终态；S04完整预算续测已通过并清理，新静态缓存共同基线也已测得收益；10/17–18先做一次独立复测、缓存生命周期与完整chunk验证，再更新profile和单位有效提交账本，同时完成质量风险收口。仅证据触发才做新数据/等预算训练；不展开cross-attention、PF2/3/4或多树搜索。最多一个成本支持的小树对照也必须独立登记和重验，n3诊断不等于已验证生产树。

10/20前条件性训练完成，10/21有限DEV，10/22明确quality+系统go/no-go；10/23冻结后进入TEST，10/26 TEST收齐、10/27获准正式性能、10/28冻结主表。10/29–31整合中文稿，11/01 18:00前交老师。主线未通过时收缩系统主张及可选环境，不增加新方向；不能保证按时形成正论文结论。

后续运行量已从实际归档核验：[预算估算](../投机采样/results/runtime/p0-wam-future-q03-q04-volume-plan-review-v1-20261016-r1/PLAN-ESTIMATE.json)。条件性Q03为6条件×20组=120局；Q04为T100+spec100+3个baseline各50局=350局，三个baseline共用事前固定的主T50子集，不额外再跑150条T。当前PCIe单worker的阶段跨度估计：Q03常规4–5.5小时；Q04均值代理12.5小时、加25%余量约15.6小时，另留1–2小时启动/CAL/归档；失败长尾可超过18小时预算。仅是未来规划，不能当SXM4性能或运行上界。小draft历史六组2000步CPU2训练约15.6分钟、六组500步约9.8分钟，完整episode才是主要墙钟开销；新数据采集和新训练均未启动。

乐观：1–2天内从少量严格病例找到可检验修复，同时缓存生命周期验证通过，10/21完成DEV。常规：先用1–2天补实际分歧证据与强baseline，随后2–3天做有限必要baseline及条件性训练DEV，10/22评审；新TF−8pp和max1.4意味着该安排有实质质量风险，并非承诺按时通过。若10/22仍无质量或B14增量证据，停止扩展，保留机制/负结果与明确缺口给老师判断。RoboTwin、LIBERO100、真机和第二架构均不占当前保证路径。

[CVPR2027官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)：注册11/10 AoE、正文11/16、补充11/23，对应北京时间11/11、11/17、11/24的19:59；当前页面没有另列abstract截止。[官方CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)需作者OpenReview资料；10/18检查、10/25处理遗留，现无作者清单，不能声称已检查。11/02起按老师反馈修订/翻译/排版，内部提交至少提前一天。

## H. 原始证据与维护

- [短索引](../.pipeline/research/speculative-sampling/WAM-COORDINATION-20261016.md)、[短状态](../.pipeline/PROGRESS.compact.md)、[Cycle](../.pipeline/cycles/wam-research-20261016/PROGRESS.md)。实际HW为已授权切换的v15，原v12/legacy保留，不再次升级/回退。
- [三档机制矩阵](../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4/MATRIX-RESULT.json)、[静态Graph](../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/graph-matrix-r1/MATRIX-RESULT.json)、[同7query顺序Graph](../投机采样/results/runtime/p0-wam-same-seven-sequential-graph-v1-20261016-r3/SEQ-RESULT.json)。监督器终态及PLAN在各自同目录。
- [机制独审及NCU](../投机采样/results/runtime/p0-wam-mechanism-results-independent-review-v1-20261016-r1)、[Q02](../投机采样/results/runtime/p0-wam-quality-q02-replay-v1-20261016-r2/RESULT.json)、[96行匹配对照](../投机采样/results/runtime/p0-wam-quality-q02-matched-cpu-v1-20261016-r1/REPORT.txt)。
- [原型源码](../投机采样/experiments/wam_candidate_tile_v1/PLAN.md)、[G2扩展数值负结果](../投机采样/results/runtime/p0-wam-candidate-tile-fixture-v1-20261016-r2/FIXTURE-RESULT.json)、[独立来源审查](../投机采样/results/runtime/p0-wam-candidate-tile-fixture-independent-review-v1-20261016-r1)。
- [v2有限数值门](../投机采样/results/runtime/p0-wam-candidate-tile-guarded-v2-20261016-r1/GUARDED-RESULT.json)、[两进程三臂合并独审](../投机采样/results/runtime/p0-wam-candidate-tile-graph-pilot-independent-review-v1-20261016-r2/TWO-PROCESS-COMBINED-REPORT.md)；原始r1/r2各目录保留所有轮，未择优。
- [实际attention counters独审](../投机采样/results/runtime/p0-wam-candidate-tile-profile-independent-review-v1-20261016-r1/R3-TERMINAL-COUNTER-REPORT.md)，原CSV与NCU report见被引用的profile r3目录；r1/r2均为未启动GPU的准备/审查修订，记录保留。
- [S04失败与原生临时槽合同核验](../投机采样/results/runtime/p0-wam-s04-independent-result-review-v1-20261016-r1/REPORT.txt)，实际job的CHUNK-RESULT/SUPERVISOR-RESULT与完整冻结源码分别保留。
- [新50对完整结果](../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/remote-evidence/fixed104/ANALYSIS.json)、[小原始证据归档](../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/COMPLETE-SMALL-TRACE-ARCHIVE.json)。六tar共3196件/60,528,640B，含全部104局结果、命令/动作、接受/阶段计数、初始相机/状态/RNG；不含大KV、obs_data或视频，不能称完整大张量轨迹。
- [四完整诊断episode视觉保全](../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/OBS-DIAGNOSTIC-PRESERVATION.json)：TF失败task1/init29及其T配对、最长共同成功task6/init25及其T配对；全部102个obs_data只含两相机uint8数组，逐件内容/名字/大小审计，七tar合计173,834,240B，52.95秒只读传回、05:14前完成。仅供后续诊断，不是FIT/CAL/TEST；没有KV/latent/权重/视频。远端原件未删，本机该dispatch所有证据共244,470,836B（约233.1MiB）。
- [50局路径score诊断](../投机采样/results/runtime/p0-tf-heldout-path-cost-descriptive-v1-20261016-r1/REPORT.md)、[全部原始质量字段独审增量](../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/RAW-COVERAGE-INCREMENT.json)。
- [AutoDL动态终态入口](../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/status.json)、[原质量协议](../.pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md)。

[最终本机作业账本](../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/NIGHT-LEDGER-FINAL-v2.json)共28个GPU/CPU准备namespace，已知GPU作业生命期3030.814秒（50.51分钟，非CUDA活跃时间），主要产物逻辑约2.77GiB。491项原tile/保护源/原记录size+mtime守卫不变，未重做整套模型hash、未删除失败、未复制逐query30层完整历史。旧派生ledger漏映射两个旧job的`own_process_reaped`字段；最终v2只修读字段，原监督器均为true，所有原job终态不改。AutoDL另计，不混入SXM4速度或本机GPU预算。

最终[报告只读独审](../投机采样/results/runtime/p0-overnight-final-local-independent-review-v1-20261016-r2/REVIEW-RESULT.json)PASS：51个本地证据链接存在、491项原保护记录不变，完整报告/短视图数字与权限一致；短入口5997B、本轮索引3279B。此处追加复核回执及Cycle本轮完成标记，不改已核验研究结论。


---

## task4有限诊断

来源：`投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/REPORT.md`。

# task4/init26 Q02 有限诊断

状态：**D01 HOLD**。本报告只读今日 50 局中预先指定的 T/TF_CONT 风险配对，以及同任务、不同初态的 TF_CONT 成功参照；不改变原 `retention=false`、FAILED/权限或明日扩大 DEV 的拒绝门。准确机器可读结果为 `RESULT-FINAL.json`；同目录早期 `RESULT.json` 和 `RESULT-CORRECTED.json` 已被它显式取代，原因是最初遗漏 `obs_data_0.pt` 的 NumPy 叶子以及实际提交深度字段。

T 在 249 个环境步成功，TF_CONT 在 796 步失败。原完整配对审计和本次复核显示：20 个保存的初始仿真/控制器数组、实际 target 初始 RNG、两路 128×128 相机像素都逐位相同。首个保存的观测帧也相同。实际环境命令从索引 0 就有 1/7 通道不同，最大绝对差 0.0078125；`actions_0.pt` 的 BF16 张量有 62 个元素不同，最大差 0.015625。第一份 `obs_data_0.pt` 中最早可见相机差异在保存帧 3。这里的保存动作张量和环境命令不是同一个字段，不能从差值直接判定哪条投机边导致失败。

TF 风险局前 14 个完整 chunk 的 252 条实际接受记录，提交深度 0/1/2 分别为 35/11/206；同任务成功参照 init35 为 33/13/206。两者的 σ 分箱计数都是 28（0.5–0.75）、32（0.25–0.5）、192（<0.25），记录的 `changed_route` 都为 0；固定阈值是 0.007839680526450805。风险局最初两条高 σ 查询深度为 0，随后有深度 1/2。相似的深度分布也出现在成功参照中，因此这组计数不支持把失败归因于“接受太多”或某个 σ 区间。参照的初态不同，只能作描述。

当前缺失逐查询 verifier 的实际 root/候选 teacher 输出与最终环境命令的无歧义对应，也没有未保存的 velocity。仅靠 cost、σ、depth、`actions_0` 和命令无法重建每条提交边的目标动作可达性，更不能判定训练方案会修复该失败。**现有证据不足以触发 D01 或更改接受阈值。**

若未来另行批准前瞻诊断，最小日志是在现有完整 episode 路径直接保存每个 verifier 查询的 frame/chunk、σ、候选边、cost、实际提交深度、已有 teacher root/候选输出的紧凑数值或可审计引用，以及映射到提交动作/环境命令的索引；不增 teacher forward、不存 30 层 KV。先用完整 DEV episode 验证该映射和日志覆盖，再决定是否采集独立 TRAIN/CAL 的新 TF on-policy 标签。额外日志估算约每完整 TF 局 0.5–2 MiB，**未实测**。本次实际 episode 墙钟 T 132.93 秒、TF 270.28 秒；若只作 2–4 对完整 DEV 病例，现有 episode 时间合计约 13–27 分钟，加启动与日志开销可预留约 20–40 分钟，均不是正式性能计时。本轮未启动这些实验。

诊断仅使用 2 个初态文件、2 份命令、各自首个保存 `actions_0/obs_data_0`、原配对与结果、以及 TF 风险和成功参照各前 14 个完整 chunk/252 条接受记录；上限是每例 20 chunk 或 256 条记录。脚本 CPU 限 2 核，实际分析小于 1 秒，CUDA 未初始化，未用 GPU、SSH、新环境或新模型 forward。

