---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: 2026-10-17T13:18:21.505241+08:00
current: M15
next: review_new_bounded_teacher_path_diagnostic_not_started
execution_status: completed_bounded_NO_GO_GPU_and_SSH_cleaned
---
# 现有 WAM 研究进度

> 当前：M15六对十二局完整，PF2投入NO_GO；C3/6、P5/6，救回2但风险仍1败/控制长度超限。GPU/SSH清理，reset全项目默认优化；四false/旧NO-GO不变。见[本轮结果](../../experiments/tf-pf2-autodl-20261017/RESULT.md)。

2026-10-17早用户明确暂停实验并要求网页版文献/开源Prompt及GitHub同步。GPU已清理，旧/新DEV共72/96完整、剩24；双根CPU报告72 raw/50严格组件/22旧TF_ON arm收据缺口，整体PARTIAL，用户暂停作为非预算completion issue保留。fresh T20、CONT19、ON20成功，MLP0完整；风险ON已完成4均失败，其中task1/29、3/26、4/26为T成功/TF失败，三对保存物理/reset/RNG/双相机exact。当前TF-on权重/配方不满足无新增风险条件，不能推荐TEST；TF64结构/API仅暂冻研究。一次新30/固定三条件500更新均已完成，不自动追加。原四权限false、FAILED/default/tile保持。当前只做CPU交接发布，见[本轮结果](../../experiments/tf-effect-closure-20261016/RESULT.md)、[暂停决定](../../memory/global/requirements/decision-暂停TF实验并先外部文献复核.md)。

2026-10-16 22:07：[TF实际接续](../../experiments/tf-effect-closure-20261016/RUNBOOK.md)。r2复制KV诊断主动停止PARTIAL；r3复用原完整OFF、ON完整50/101请求/2192forward/904树/2712行及RESET/final全KV、逐chunk RNG核验完成，原45分钟内清理。CPU独立2712行23.8秒通过。新30本机root3672051/clock3672054正在运行（4h/2GiB），尚未新训练/DEV；未来DEV96清单已冻、按首条270.5秒任务前瞻预算7h/总12h。全部旧FAILED/四false/默认/tile不变；原双新pass/中间KV门不冒称PASS。

2026-10-16 21:18 用户已授权[TF收口1→2→3](../../experiments/tf-effect-closure-20261016/EXPERIMENT.md)：新CPU诊断完成，当前到达parent同step误差较target FIT高约48%，联合覆盖20.52%超过旧CAL对应P95，但与旧mixed误差接近，不能称严重偏移/失败根因。支持唯一一次新数据对照。完整记录50chunk采集OFF/ON资格r2实际运行（root3647250/clock3647251），r1父PID准入FAILED/0GPU保留，共用原45分钟帽。新30FIT/CAL、训练、DEV尚未启动。原四权限/默认/FAILED不变，AutoDL不连接。

M13 TF优先CPU覆盖审计已完成：[2712真实query](../../experiments/tf-training-support-20261016/EXPERIMENT.md)。未发现足以宣称严重递归尺度偏移的证据；结构固定TF64，继续核监督/分层误差，训练不强行触发。此条替代下方M12已完成后的“数据核验待开始”，不改变原FAILED/四权限。

2026-10-16T19:54:50.062514+08:00 用户进一步固定TF64为主draft，MLP仅baseline、算子后置。M12两真实记录chunk与同query/时间反事实完成，根因边界见[结果](../../experiments/draft-teacher-path-20261016/RESULT.md)；11.72分钟/约0.45GiB、自有进程清理。训练尚未触发，下一步核TF训练来源与递归监督，旧四权限false、120未启动。

外部评审：代码与关键HW已发布`qingyiyi/WAM-egale@b0dfac3a`，[入口/完整prompt](../../research/speculative-sampling/GITHUB-REVIEW-EXPORT-20261016.md)。网页执行方案已返回；本次固定此 commit 继续，原研究准入不变。

M11有限诊断终态PARTIAL：[五报告/时间表](../../experiments/task4-direction-20261016/EXPERIMENT.md)。T两次249成功、TF一次796失败，第二TF预算中断；1对完整而非原定2对。原r1实现FAILED/r2预算FAILED均保留，累计54.98分钟、全部自有worker/guard清理。首teacher分歧早于draft提交，完整r0链路/初态核验通过；不触发训练，四权限false、120不启动。下一步同真实query及完整KV数值复验，研究仍active。

当前：M10的50完整DEV已结束，TF保留条件未过；研究active。结构接口暂冻供算子，权重/配方未accepted；未来120门实际拒绝，无注册/CPU waiter/GPU。最新依据见[今天结果](../../research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)。下方夜间/晨间结论为各自历史范围，晨间扩大建议superseded_by今日HOLD。

## 夜间有限工作结论（晨间更新见后文）

夜间有限工作已执行完，完整报告和短入口已更新，最终独立报告复核PASS。研究整体尚未完成。新AutoDL104/104完成并清理：T48/50、TF44/50，差−8pp；共同成功43局长度比P50/P90/max为.996/1.054/1.4。实际初始相机47/50严格一致；物理/控制器/RNG50/50一致。当前TF质量门false，正式45chunk仍受阻，旧A_off默认与原FAILED保持。

真实M32→224已落到库GEMM；同七query批处理经济性成立。路线B数值/状态重放六完整policy/cache chunk、300提交通过；共同文本缓存下静态30层新B14相对原B14短/中/长下降3.86%/6.45%/18.40%，仅一个进程固定输入pilot，尚无缓存生命周期、完整chunk性能或闭环收益。原tile未动，普通缓存不作为新创新。

## 完整计划状态

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
|---|---|---|---|---|
| `M1` | 原本机闭环终态 | completed | fresh30及parent10原结果、像素FAILED均保留 | 不借旧证书放行新TF |
| `M2` | AutoDL及接续 | completed_research_quality_failed | 旧100原FAILED；新104完整研究终态05:04、清理通过；小档及四局完整视觉归档 | 实例未关闭；不续跑 |
| `M3` | 网页计划核对 | completed | 主线修正、边界和日期已核对 | 最新事实覆盖旧快照，保留历史 |
| `M4` | 真实机制 | completed_bounded_window | 三KV全30层、Graph、同7query顺序对照完成 | 补新共同强基线复测及实际有效提交账本 |
| `M5` | 质量风险与隔离 | completed_bounded_window | Q02有限重放、同类历史candidate对照、全50局路径日志审计完成 | D01/D02 HOLD；严格病例与必要简单baseline |
| `M6` | 有限系统原型 | completed_bounded_window | 两进程未缓存pilot、NCU、六完整chunk状态门、单进程共同缓存pilot完成；旧失败保留 | 独立复测、动态生命周期、cached完整chunk与新profile |
| `M7` | 晨报与证据 | completed_bounded_window | 最终独审PASS：51本地链接、491保护记录、数字与边界一致 | 夜间结果回传；研究整体仍active |
| `M8` | 晨间TF固定补训 | completed_research_only | CONT固定500更新及原训练记录保留 | 不借训练结束签质量证书 |
| `M9` | 晨间80完整DEV | completed_research_only | CONT20/20、主长度通过；原聚合PARTIAL/像素问题保留 | 不借旧描述签证书 |
| `M10` | 等预算MLP与前瞻50完整DEV | completed_retention_hold | fresh三臂10/10；风险task4新增失败，retained=false；全raw/来源/清理独审 | 当前120门拒绝 |
| `M11` | 风险病例前瞻日志诊断 | completed_bounded_partial | 1对完整风险复现、teacher早于draft偏离；第二TF预算中断 | 同query/全有效KV数值复验，训练HOLD |
| `M12` | 同query教师路径定位 | completed_bounded_window | 两真实记录chunk与时间编码反事实完成，11.72分钟/约0.45GiB清理 | 数值差异不等于单一失败因果，见M13有限监督实验 |
| `M13` | TF64一次数据配方收口1→2→3 | completed_bounded_partial_quality_NO_GO | 新30/固定训练完成；DEV77/96；TF fresh20/20、risk4/8对T6/8，四false | [最终结果](../../experiments/tf-effect-closure-20261016/RESULT.md)，先核网页方案，不自动新训练/TEST |
| `M14` | 冻结TF_ON/PF2只读审查 | completed | 源证据/选定权重/数据边界、双A100资源、六pair建议与未应用diff | [只读结果](../../experiments/tf-pf2-plan-20261017/RESULT.md)；待新具体批准，不自动开跑 |
| `M15` | 冻结TF_ON/PF2双卡完整DEV | completed | 六对十二局完整/清理，投入NO_GO；旧四false保留 | [结果与建议](../../experiments/tf-pf2-autodl-20261017/RESULT.md)，新诊断尚未登记/启动 |

## 作业终态与边界

本机28个GPU/CPU准备namespace已知GPU作业生命期合计50.51分钟，实际GPU作业均清理。S04失败/超时和其他失败均记账，不拼接通过片段冒充完整通过。491项原tile/保护源/记录size+mtime守卫不变。AutoDL原4小时截止05:19，比用户09:00授权上限更早；未自动扩预算或续租。所有实验worker已退出，租卡实例未关机。原始小档60.5MB、另四完整诊断episode相机档173.8MB，本机dispatch共233.1MiB；没有删远端或失败原件。

## 下一步

10/17–18并行收质量风险与强系统基线；只有证据触发才一次新数据/等预算训练。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文稿交老师。质量、最终独立TEST、非劣和正式性能许可四项分开，当前均false。研究计划及具体证据见[短索引](../../research/speculative-sampling/WAM-COORDINATION-20261016.md)和[夜间结果](../../research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)。

最终复核时间：2026-10-16T05:27:50+08:00。本轮完成标记仅关闭夜间有限工作，质量门不变。见[最终只读独审](../../../投机采样/results/runtime/p0-overnight-final-local-independent-review-v1-20261016-r2/REVIEW-RESULT.json)。

## 晨间有限工作 M8/M9 已完成，研究仍 active

TF_CONT同child23固定额外500更新完成；两站四臂80完整DEV均结束。T19/20、child19/20、CONT20/20、MLP20/20；CONT/T19条共同成功步数比P50/P90/max=1.004184/1.032907/1.172996，六数值描述门通过。物理/实际RNG60/60，相机56/60，主CONT19/20。

本机r3异常消失，原13+中断保留、原因UNKNOWN；r4同40全量恢复不拼接旧13，继承原3h/3GiB。远端r1环境失败0局，r2完整40但历史凭证文件名触发原FAILED/无RESULT；冻结聚合实际PARTIAL，独立80描述不放行原质量。固定13恢复审计commands10/13、success12/13一致，T task8旧393成功→新796失败且初图不同，不能归因。四质量权限、旧默认和原TF正式45chunk限制不变。

本机/远端全部注册自有worker与guard已退出；累计记账3.47674h含失败/等待/重试间隔，非CUDA活跃时间，各站不超3h。两本机质量目录1.923GB<3GiB。AutoDL交还用户关机，不再SSH/启动；实例关闭未核实。原FAILED/原tile/保护状态文件未改。

暂冻TF64和固定CONT权重仅作扩大验证候选。先核评测重复性、初始配对与布局预检，前瞻登记必要完整重跑/独立测试；新FIT/CAL HOLD。系统并行补共同缓存强baseline的复测/生命周期/完整chunk，不新增大融合。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文稿交老师。

详细证据：[TF本轮结果](../../research/speculative-sampling/TF-USABILITY-RESULT-20261016.md) · [本轮PLAN](../../../投机采样/results/runtime/p0-tf-usability-v1-20261016-r1/PLAN.md)。本次完成只关闭有限验证，不代表整体研究完成或质量PASS。

## 午间 M10 已完成，权重/训练配方未验收

50完整horizon800已收尾：fresh init35 T/TF/MLP各10/10，TF长度P50/P90/max=1.021212/1.074884/1.092593。risk各5/8但task4/init26 T249成功、TF796失败，原事前risk_no_new_TF_failure=false；反向task0救回不抵消。28保存物理/controller/实际初始Torch RNG一致，相机25/28；否决pair双相机也exact。

3184PT/34776数组及50份commands全量独审一致；173源/两权重/prereg/491保护记录不变。实际GPU作业8056.511秒、2.207GB、61859计数均未越预算，58自有pins退出。AutoDL已归还不再SSH，关机未核。四权限/default/旧FAILED不变。

TF64结构和7状态B14接口暂冻作算子研究；3000权重与训练配方没有通过保留条件。120局源已准备，但真实前置实际拒绝，未注册或启动waiter/GPU，不能改旧RESULT放行。TEST40–49仍待候选/引擎冻结。

M11只做有限前瞻日志诊断计划：task4首命令已异，但现有深度/sigma计数不能定位失败；缺teacher输出到环境命令映射，D01 HOLD。拟2–4对完整病例运行20–40分钟，入口实现/审查另0.5–1.5h；尚未登记/运行。用户质量优先、算子正式计时通知后完整30局分片末暂停的偏好保留。

证据及日历统一见[今天报告](../../research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)，不在本Cycle复制长计划。11/01 18:00中文稿交老师不变。
