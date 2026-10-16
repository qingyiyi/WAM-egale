---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: 2026-10-16T15:18:27.460014+08:00
current: M11
next: prospectively_log_teacher_to_command_before_any_data_or_expansion
---
# 现有 WAM 研究进度

当前：M10的50完整DEV已结束，TF保留条件未过；研究active。结构接口暂冻供算子，权重/配方未accepted；未来120门实际拒绝，无注册/CPU waiter/GPU。最新依据见[今天结果](../../research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)。下方夜间/晨间结论为各自历史范围，晨间扩大建议superseded_by今日HOLD。

## 夜间有限工作结论（晨间更新见后文）

夜间有限工作已执行完，完整报告和短入口已更新，最终独立报告复核PASS。研究整体尚未完成。新AutoDL104/104完成并清理：T48/50、TF44/50，差−8pp；共同成功43局长度比P50/P90/max为.996/1.054/1.4。实际初始相机47/50严格一致；物理/控制器/RNG50/50一致。当前TF质量门false，正式45chunk仍受阻，旧A_off默认与原FAILED保持。

真实M32→224已落到库GEMM；同七query批处理经济性成立。路线B数值/状态重放六完整policy/cache chunk、300提交通过；共同文本缓存下静态30层新B14相对原B14短/中/长下降3.86%/6.45%/18.40%，仅一个进程固定输入pilot，尚无缓存生命周期、完整chunk性能或闭环收益。原tile未动，普通缓存不作为新创新。

## 完整计划状态

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
|---|---|---|---|---|
| M1 | 原本机闭环终态 | completed | fresh30及parent10原结果、像素FAILED均保留 | 不借旧证书放行新TF |
| M2 | AutoDL及接续 | completed_research_quality_failed | 旧100原FAILED；新104完整研究终态05:04、清理通过；小档及四局完整视觉归档 | 实例未关闭；不续跑 |
| M3 | 网页计划核对 | completed | 主线修正、边界和日期已核对 | 最新事实覆盖旧快照，保留历史 |
| M4 | 真实机制 | completed_bounded_window | 三KV全30层、Graph、同7query顺序对照完成 | 补新共同强基线复测及实际有效提交账本 |
| M5 | 质量风险与隔离 | completed_bounded_window | Q02有限重放、同类历史candidate对照、全50局路径日志审计完成 | D01/D02 HOLD；严格病例与必要简单baseline |
| M6 | 有限系统原型 | completed_bounded_window | 两进程未缓存pilot、NCU、六完整chunk状态门、单进程共同缓存pilot完成；旧失败保留 | 独立复测、动态生命周期、cached完整chunk与新profile |
| M7 | 晨报与证据 | completed_bounded_window | 最终独审PASS：51本地链接、491保护记录、数字与边界一致 | 夜间结果回传；研究整体仍active |
| M8/M9 | 晨间TF固定补训与80完整DEV | completed_research_only | CONT20/20、主长度通过；原聚合PARTIAL/像素问题保留 | 不借旧描述签证书 |
| M10 | 等预算MLP与前瞻50完整DEV | completed_retention_hold | fresh三臂10/10；风险task4新增失败，retained=false；全raw/来源/清理独审 | 当前120门拒绝 |
| M11 | 风险病例前瞻日志诊断 | planned_not_registered | teacher到命令映射缺失；D01 HOLD | 独立入口/预算/完整episode/无新teacher调用 |

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
