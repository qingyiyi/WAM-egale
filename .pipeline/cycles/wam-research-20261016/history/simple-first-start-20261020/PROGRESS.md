# 当前研究进度

更新：2026-10-20T21:59:48+08:00。Cycle：wam-research-20261016；[PLAN](PLAN.md)。current M39，next M39 逐阶段三路径对齐；M40 GPU1固定42采集自动接续。

|ID|状态|当前结果|
|---|---|---|
|M17|in_progress|四套6000局计算齐，科学审计未放行|
|M31–M38|completed|原基线、profile、子图与正负结果保留|
|M39|in_progress|当前context128完整层开销分析完成；数学源码未改；下一步先逐阶段原版/独立/嵌入对齐，再改主要差距与ready调度|
|M40|in_progress|现场10/20 22:07:29北京：新增6/42完整终态、5成功1完整算法失败（open_microwave），当前`place_object_basket-valid02`；原首六5成功1失败保留。|

## 当前算子证据

[本轮分析](../../experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)：24/24完整block、150/150 Torch Linear，有限输出/中间量/有效KV逐位通过。三KV插桩阶段GEMM占57.26/53.02/48.96%；cross K/V为其中1.88–2.02ms。cross-K RMS约0.606–0.654ms，对应native单kernel0.068–0.072ms；长KV self+cross attention占23.02%。阶段含队列/等待，native只kernel时间，不能混作正式算术吞吐比。

当前108CTA×256线程、255register、98848B shared、1608B localSizeBytes。context tile128×128/K64、action64×128；4compute warp做CUTLASS/FA2，8warp协作GEMM epilogue。pointwise2048元素仍仅4warp运算；Norm一CTA一行，shared转换和单warp写回。阶段grid.sync仍在，非完整跨阶段ready执行器。NCU查询ERR_NVGPUCTRPERM，动态occupancy/spill/HBM/L2 UNKNOWN，不绕过权限。

[上一轮普通计时](../../experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESULT.md)仍为最新速度：B14短/中/长同轮native比1.866–1.868 / 1.836–1.875 / 1.732–1.737。uncached eager、准备排除，不是强cached/Graph、30层、chunk/E2E或闭环质量。本轮插桩不替代普通计时；旧ready负结果不改。

[资源](../../experiments/single-layer-async-gemm-20261020/context128-profile/RESOURCE-AND-STOP.md)与[终态审计](../../experiments/single-layer-async-gemm-20261020/context128-profile/FINAL-AUDIT.json)：约23.2/900秒GPU生命周期、约25.3MiB证据，owned结束，GPU0无算子后继队列。源码/所选二进制核验保留；旧1497/1500与372/420、FAILED和原截止不改。

## RoboTwin接续

[新固定42队列](../../experiments/robotwin-pilot-20261020/target-remaining42/PLAN.md)，现场10/20 22:07:29北京：新增6/42完整终态、5成功1完整算法失败（open_microwave），当前`place_object_basket-valid02`；原首六5成功1失败保留。本次最新明确继续为新阶段授权，旧截止不重写。21:37:02启动、次日03:37:02硬截止，含清理6小时帽；每局1800秒/8000teacher，开局留整局及900秒清理，4GiB新增存储，累计原24GPU小时帽。

首六监督4150实际query已核对；新增30FIT/12CAL，合并成Target48（FIT36/CAL12）。复用108完整512-token BF16 bank；完整episode cached/uncached等价尚未测。GPU1单worker、server/client各2CPU线程，固定轮转自动接续，不重试infra失败；自然算法失败保留。状态见[LIVE-SNAPSHOT](../../experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)。完成后需审计并实现后续TF适配/训练/DEV，本guard不自动训练。采集结果不是TF成功率。

四套原LIBERO T/C/P各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。首两S审计FAILED、训练/DEV曝光和长尾保留，非独立TEST。

## 下一步与不变边界

最新用户反馈：先逐阶段原版/自有独立/自有嵌入整层对齐，再优化Norm/pointwise、FA2等主要差距，最后context K/V ready填尾波。归约顺序、BF16舍入、合法KV不放宽；一次一个因素。普通缓存不作为MegaTransformer新贡献；最强1/3/7×cache矩阵仍缺。

四项engineering_quality_screen/independent_test_complete/noninferiority_established/formal_performance_permission均false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保留。原始action沿原算子；baseline/投机代码/原tile/HW/state/cycle/rules不动。BAAI只写~/cxz，AutoDL归还，不改驱动/时钟/服务或杀他人。仅main发布源码/HW/小证据。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。此前[完整进度](history/context128-profile-start-20261020/PROGRESS.md)保留。

最新[逐阶段对齐计划](../../experiments/single-layer-async-gemm-20261020/context128-profile/STAGE-ALIGNMENT-PLAN.md)尚未启动新GPU；当前profile预算已终态。
