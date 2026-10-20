# 当前研究进度

更新：2026-10-20T19:25:00+08:00。Cycle：wam-research-20261016；[PLAN](cycles/wam-research-20261016/PLAN.md)。当前 M39 本轮完成，next M39 当前profile/ready设计；M40保持待TF迁移。无GPU后继队列。

|ID|状态|当前结果|
|---|---|---|
|M17|in_progress|四套6000局计算齐，科学审计未放行|
|M31–M38|completed|基线、profile、原子图与正负结果保留|
|M39|completed|本轮较大tile有限通过；B14整层三KV均≤同轮native两倍，尚未达到等速|
|M40|in_progress|RoboTwin六FIT Target已5成功1失败；TF采集/训练/DEV未齐|

## 当前算子

[较大tile结果](experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESULT.md)：仅optimized增加8warp协作epilogue、2048元素point/cache任务、context128×128/K64（action保持64×128）。两variant各9真实签名/每格23中间量/18有效KV转移、重复输出/KV、输入/RNG/tile逐位通过。仍整个单层一个persistent kernel。

所选128版B14短/中/长的两个独立进程耗时比1.866–1.868 / 1.836–1.875 / 1.732–1.737；完整层延迟分别约4.898–4.914 / 4.677–5.168 / 5.095–5.247ms。uncached eager、准备排除，每格每引擎16普通样本；profile独立，每条1compute kernel。单层工程门，不是强cache/Graph、30层、chunk/E2E或机器人质量。中KV绝对波动原样报告，不混合中位掩盖。

资源108CTA/256线程、255register、98848B shared、1608B localSizeBytes。阶段grid.sync仍在；旧FFN ready本轮未重验、defaultFalse；动态occupancy/spill/流量UNKNOWN。大tile增加共享内存，不能说已经解决资源瓶颈。

[资源](experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESOURCE-AND-STOP-REPORT.md)：新372/420block、GPU worker生命周期69.379/900秒，两CPU构建约238.5秒，guard约308.3秒；不重置旧1497/1500、原FAILED与20:20:28截止。六GPU worker与两build结束；最终[审计](experiments/single-layer-async-gemm-20261020/larger-tile-followup/FINAL-AUDIT.json)。旧[2.54–3.02倍版](experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)和旧ready负结果保留，不把跨轮数字作为严格新旧AB。

## RoboTwin与质量

[现场检查](experiments/single-layer-async-gemm-20261020/larger-tile-followup/ROBOTWIN-LIVE-CHECK.json)：首六FIT身份Target 5成功，open_microwave完整失败；83policy/6557teacher。失败实际1520命令/罚分1500分列；初始物理和观察逐位。114专家初态、108完整BF16 bank齐，完整任务cache等价未测。所有owned闭合，无后继自动队列，TF collector/训练/DEV尚未实现齐；本轮关注终态，不按过期首六局链重启。

[四套原LIBERO](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499（T/C/P各500）。首两S审计FAILED、训练/DEV曝光与长尾保留，非独立TEST；旧330/360计时不签发新算子权限。

## 下一步与边界

以context128为研究实现，先做当前版本阶段/寄存器spill/调度成本定位，再将独立context K/V及norm变成ready任务填尾波；consumer必须等真实依赖完成。阶段同步没有全局去除。保持原始action用原算子，不开30层或生产集成；baseline/依赖、投机代码、原tile/HW、state/cycle/rules不动。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保持。BAAI写入仅~/cxz，AutoDL已归还，不改驱动/服务/时钟、不杀他人。

只main同步源码/HW/小证据，不公开模型/PT/DSO/私有SASS/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。此前完整记录见[历史](cycles/wam-research-20261016/history/larger-tile-followup-start-20261020/PROGRESS.md)。
