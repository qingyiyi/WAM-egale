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

[结构接口](../../../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/OPERATOR-INTERFACE.STRUCTURE-FIXED.json)明确只作算子研究载体，权重/配方未accepted。

TF53070参数：16token×17特征，state7、previous-direction7、sigma、next-sigma、t/1000；根使用已有方向，递归使用父draft方向，不能统称previous target。单层pre-norm、D64/H4/FF256、GELU、dropout0、学习位置、两组7维残差头；没有图像/prompt cross-attn，整个WAM仍有视觉条件。MLP52928参数作等容量对照。小模型参数与内部特征FP32，adapter返回current_state.dtype，本轮为BF16；接受打分FP32。

固定seed20260923/final3000，TF权重SHA3657388f…d64ac，匹配MLP为3dc01a48…103b0。训练不是连续3000更新：

1. FIT174 target159691 query，2000 updates、batch512、AdamW lr1e-4/wd.01；任务→episode→行均衡抽样，FIT-only固定normalization。
2. 500 updates，target256＋历史candidate256、lr3e-5/wd.01，重置AdamW，不改norm。
3. 同一起点2500再固定500，重置AdamW，重复原500抽样表，数据/比例/lr/norm不变。TF CPU125.154秒；匹配MLP CPU54.622秒。

历史candidate14997行=3906 A_off＋11091旧TF，不能称当前TF的on-policy数据。损失、来源及完整SHA见接口；没有根据这次DEV挑seed/checkpoint。此次结果未证明增加训练步数是修复方法。

verifier保持K2/B2 bonus、7状态×CFG2=B14、Q16/状态、prefix_then_cost、共享历史/O3、旧sigma加权L2阈值0.007839680526450805；PF关闭。reset仍共同使用512-token BF16 bank。本轮没有集成新算子或文本投影缓存。

## task4有限诊断：根因仍未知，D01 HOLD

[最终诊断](../../../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/REPORT.md)核20个初始sim/controller数组、target RNG及双相机exact。环境命令索引0已有1/7通道差异、max.0078125；保存actions_0有62个BF16元素不同、max.015625。第一份obs_data_0最早相机差在保存帧3；保存帧不是已证明的同编号环境步，不能直接判某条边是原因。

失败局前14完整chunk、252条记录，提交深度0/1/2为35/11/206；同task成功init35为33/13/206，sigma分箱相同。记录的changed_route=0仅描述已有选择器日志，不是重新验证packed/sequential数值等价。成功参照初态不同；相似计数不能证明安全，也未给出“接受太多/某sigma/训练分布”根因。

现有日志缺逐查询实际teacher root/候选velocity及到最终环境命令的无歧义映射，不能补算未保存数据。因此没有足够证据触发D01、累计误差预算、PF或cross-attn；这些继续HOLD。

下一作业应先在独立入口前瞻登记2–4对完整诊断episode，直接保存已经计算的teacher输出、sigma/边/提交索引及到环境命令的映射，限制最多20完整chunk或256查询的详细日志，不增加teacher调用、不存逐query30层KV。按本轮T132.93秒、TF270.28秒的实际病例墙钟，2–4对纯episode约13–27分钟，运行可留20–40分钟；入口实现/审查另估0.5–1.5小时。新增紧凑日志容量尚未实测，需设明确cap。此作业尚未登记/启动，不是smoke，也不能冒充独立TEST。

## 系统线与放行关系

H1有真实M32/96/224及attention Q16账本，原始瓶颈机制尚需补强；H2同7冻结query的批处理经济性成立，不能称机器人加速。H3现有路线B有强B14上的有限增量：未缓存两进程全档下降4.98%；共同文本投影缓存单进程pilot下降10.27%，尚缺独立复测、生命周期与cached完整chunk。H4当前TF未收口。本轮没有新增这些系统测量；原始证据见[夜间结果](OVERNIGHT-RESULT-20261016.md)。

明天算子侧可沿独立命名空间、固定7状态B14接口继续：先复测最佳共同缓存基线并验证动态生命周期，再做真实短/中/长KV的block/30层/完整chunk状态与有效提交成本；原tile与撤回融合保持。优先已有路线B，不启动30层大融合。最终候选权重/引擎/独立TEST仍各自冻结，不借用结构接口作质量证书。

明天init36–39的120局扩大DEV为4×30完整任务；源码/181来源绑定/clock/有界dispatcher/控制器均已审查。真实_today_gate已只读验证拒绝，当前无REGISTRATION/CPU waiter/GPU分片。下一候选需要新的前瞻版本，不能修改旧retained=false放行。TEST40–49未运行/未读结果，留待候选与引擎冻结。用户偏好持续：先质量验证，需要算子正式计时时通知在完整30局分片末暂停；正式计时需无争用。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission四项均false；全局旧A_off不变，TF正式45chunk入口仍受阻。

## 关键路径与止损

默认主线：补齐风险病例证据→仅证据触发一次新FIT/CAL与等预算训练→有限DEV→候选/引擎冻结→独立TEST→质量获准后的正式性能。TF结构暂冻，架构/阈值/PF不继续网格搜索。

乐观：下一诊断能定位可修字段，10/18前恢复质量收口；常规：一次有限数据/训练至10/20–21，与系统复测并行；不利：10/22仍不收口就停止新增研究臂，保留负结果并收缩完整机器人加速主张。三者是排程情景，不是已跑作业或保证。10/23冻结、10/26测试、10/27获准计时、10/28主表均依赖前门；11/01 18:00完整中文稿交老师不变。第二环境/真机不展开。

10/16重新核[CVPR官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)及[CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)：论文注册11/10、正文11/16、补充11/23 AoE，分别北京时间11/11、11/17、11/24 19:59。两页未另列abstract截止，不推断注册字段；所有作者需OpenReview profile，无机构邮箱的新账号审核可长达两周。10/18前检查profile、10/25处理遗留；本轮未发外部消息或改账号。

## 原始证据导航

- [今天PLAN/预算与动作](../../../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PLAN.md)、[原预登记](../../../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PREREGISTRATION.json)。
- [原50报告](../../../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/REPORT.md)、[原完整RESULT](../../../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json)。
- [全50原始独审](../../../投机采样/results/runtime/p0-draft-freeze-full50-independent-raw-audit-v1-20261016-r1/SHORTREPORT.md)、[来源/预算/真实清理独审](../../../投机采样/results/runtime/p0-draft-freeze-terminal-independent-review-v1-20261016-r1/REVIEW-RESULT.json)。
- [有限诊断最终JSON](../../../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/RESULT-FINAL.json)、[未来入口被阻塞状态](../../../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/FUTURE-VALIDATION-READY.json)。
- [晨间80与原PARTIAL](TF-USABILITY-RESULT-20261016.md)、[历史系统/质量负结果](OVERNIGHT-RESULT-20261016.md)、[原统计协议](STATISTICAL-QUALITY-PROTOCOL-20261015.md)。
