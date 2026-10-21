---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
status: active
updated: '2026-10-21T12:25:51.468246+08:00'
---
# 当前研究进度

|ID|阶段|状态|证据/下一步|
|---|---|---|---|
|M17|LIBERO|in_progress|6000局齐；保留曝光/首两S/长尾缺口，未独立TEST|
|M39|单层对齐|in_progress|Cross KV/FFN两项完成，整层再下降9.46%–10.87%；仍慢15%–31%，先FFN原语，再ready|
|M40|RoboTwin|in_progress|GPU1 CURRENT18→审计→ON/CAL；DEV90入口待审查，尚未排队|

## M39 当前结果

[完整报告](experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md) / [全部分轮与两进程](experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/FULL-LAYER-ALL.csv)。
固定数学、tile、阶段同步，仅把Cross KV和B14 FFN的tile循环放入专用阶段设备函数。Cross增量8.84%–10.60%，FFN再改善0.587%–0.700%。直接最初版/最终版对照如下，不相乘分轮倍数：

|KV|native ms（两进程）|最终 ms（两进程）|较本轮前下降|仍比native慢|
|---|---|---|---|---|
|288|2.1197/2.1207|2.7668/2.7663|10.84%–10.87%|30.44%–30.53%|
|1152|2.4052/2.4051|3.0268/3.0285|9.99%–10.13%|25.85%–25.92%|
|2160|2.8461/2.8441|3.2809/3.2809|9.46%–9.59%|15.28%–15.36%|

GPU0无别的compute context、GPU1机器人并行；固定60预热/交错12测量，观测SM1410MHz。uncached/eager完整单层，不是强cached/Graph、30层、chunk或E2E。
两阶段各9真实签名、23中间量、18有效KV转移及输入/RNG有限逐位通过，输出max_abs=0。单kernel；不外推所有输入或闭环质量。

新同边界诊断：Cross K/V约792/787→609/607µs，FFN1约236→214µs、FFN2约167→159µs；三KV都测。插桩影响整体约±1%，阶段差额不可相加替代未插桩总收益；旧全29阶段表保留为旧版本。
源码与SASS支持旧每tile进入通用函数、新每CTA每阶段进入一次，减少反复函数保存/恢复的机会；不量化动态spill。寄存器255、shared98848B、静态1CTA/SM仍在；local1472→1456B不是动态流量。NCU动态occupancy/spill/HBM/L2 UNKNOWN。

本轮GPU worker生命周期120.995秒、CPU准备/编译523.226秒，M39累计1770.668/36000秒。原14:00:42截止、旧失败/预算保留；12:19所有本轮owned退出、GPU0释放。当前仍阶段grid.sync，未ready。保留Cross/FFN修改，不恢复负收益FFN2并行K；下一步先FFN主循环/分段归约/epilogue与整层资源差距。

[前轮小算子与CTA等待](experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md)、[同前序诊断和首失败](experiments/single-layer-async-gemm-20261020/gemm-context-control-20261021-r2/RESULT.md)及[本轮前完整状态](cycles/wam-research-20261016/history/gemm-function-control-20261021/PROGRESS.md)保留。早到CTA等待与别的CTA计算重叠，不等于纯barrier时间或可兑现收益。

## M40 持续运行

[接续登记](experiments/robotwin-pilot-20261020/current18-20261021/RESULT.md)；11:32在GPU1启动CURRENT18，12:26:32检查12/18完整、9成功，place_object_basket/valid13仍执行。此为FIT/CAL采集，不是DEV成功率。108完整512-token BF16 bank沿用；RoboTwin完整episode reset等价未另测。
自动watch等待完整18及清理，再审计→ON3000→接口CAL；DEV90没有排队。CURRENT18新截止18:32:41，后继最晚10/22 05:30；原24h累计预算和旧截止保留，开局前累计5.7624h。
Target48、PARENT18（13成功5自然失败）、CHILD2500/CONT3000与10771行部署CAL已完成；旧训练解析失败0更新、valid08原PARTIAL及补跑自然失败保留。CAL不等于teacher路径或闭环质量通过。

## 共同边界

LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。不是独立TEST；paused72、旧默认A_off、retention/FAILED与四质量/性能权限false不变。
每子算子按合适tile与自然任务数，不凑108；未来按真实DAG就绪填空。只改optimized，保护baseline/原tile/LIBERO生产与state/cycle/rules。
BAAI所有写入仅~/cxz，不改驱动/服务/时钟；不杀他人。GitHub仅main小证据，无权重/PT/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包。
