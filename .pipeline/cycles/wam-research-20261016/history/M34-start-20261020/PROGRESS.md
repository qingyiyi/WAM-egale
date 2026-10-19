---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T22:56:41.237773+08:00'
plan: PLAN.md
current: M35
execution_status: isolated_engineering_measurement_running
next: M34
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

## M35 — 最新实际执行

用户新允许统一远端性能；已核对四套6000局，成功率Long T483/C484/P473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各500。新独占工程计时已启动GPU0 Goal、GPU1 Spatial，Object/Long自动后继，各套10task/init49×三臂×3独立server重复，共最多360局。新重复不充当独立质量样本，旧四false不改。预算和完整证据见[M35](../../experiments/libero-four-suite-performance-20261019/ISOLATED-MEASUREMENT-PLAN.md)。每卡一个worker，无外部compute争用观察；实际终态以队列/guard文件为准。

最新用户允许现在并行编码/编译M34，对应卡计时结束后再做GPU验证：整层单kernel＋warp职责分工，参考MegaMoE；M36只调研RoboTwin，不启动第二环境。

## M33 — 最新交付；M34整层实现待完成

[M33实际profile](../../experiments/single-block-persistent-profile-20261019/RESULT.md)完成。B14长KV静态/阶段内persistent均34 kernel，增量207.072us由self/cross attention与FFN1合计207.2425us解释。108 CTA上限低于理论2 CTA/SM容量；隔离216 CTA反事实使B14短中长相对静态退化从4.154%/4.781%/5.459%降到0.726%/0.474%/0.179%，没有新增加速。不锁时钟，跨case绝对时间非单调，不能作正式性能。

九真实block×四模式两次profile（153调用）、有效三B14反事实112调用，输出/有效KV/RNG通过。NCU权限失败，实测occupancy/L2/DRAM UNKNOWN；两个扩展身份/符号实施失败保留，第三attempt隔离后完成。五GPU job生命周期81.856秒、293 block尝试含失败、0full30/0episode；22:53两GPU无进程，全部owned退出，新证据42.91MiB。baseline及依赖88项不变。

用户要求的是**整个单层统一persistent kernel，内部warp组调度全部tile**。当前M32只有attention/FFN1阶段内常驻循环，QKV/Wo/FFN2仍host cuBLASLt，不满足整层目标。[完整DAG与缺口](../../experiments/single-block-persistent-profile-20261019/WHOLE-LAYER-EXECUTOR-GAPS.md)已登记，整层实现仍待完成。下一M34补设备原语、ready/生命周期与统一执行器，不扩30层，不以新wrapper或多kernel包装宣称完成。

M32原证据：9block/90有效KV/48原语；两进程378样本，新增persistent对静态B14慢4.42%–6.26%、B2慢2.56%–2.80%；A/A最大10.07%保留。原源码/DSO与此轮216诊断副本分列，[M32结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)不改写。

## 既有证据与质量线

M31独立静态子图9block/72状态及独立进程54状态是旧证据；M28六actual30/动态block性能是旧版本，不继承到新M32。M21/M25/M26 FFN2/NCU/M27与旧守卫失败保留。完整历史见[归档](history/M32-before-summary-20261019/PROGRESS.md)。

真实D3072/H24/d128/FFN14336；action Q16、Linear M32/224，每attention实例Q16；self-K288/1152/2160，video本批272/1136/2144，cross512。M29文献近邻主张门与M30真实DAG仍有效。

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500；[成功率表](../../experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。描述性覆盖，不解除首两S严格配对FAILED、训练/DEV曝光、组件审计与长尾。

Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

TF64/TF_ON/normalizer/tau/K2B2/bonus/prefix_then_cost与BF16/FP32冻结。PF2=SST完整policy，T50更新/51调用含末尾；reset/cache不计周期。所有臂同用512-token BF16 reset bank，cold另列；actual796与800罚分分列。

## 下一步与边界

M34补齐设备GEMM及全层ready执行器；M35两卡计时仍在运行，算子GPU验证等待对应卡释放。只做一层，最新用户允许后续性能统一远端。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

用户最新澄清：要单层统一persistent执行器内的tile调度。M32只是局部原型；M33实际profile已登记，统一整层尚未实现。

## 计划项状态索引

| ID | 状态 |
|---|---|
| `M17` | in_progress（计算完整、审计未放行） |
| `M31` | completed |
| `M32` | completed |
| `M33` | completed |
| `M34` | in_progress |
| `M35` | in_progress |
| `M36` | planned |
