---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T21:08:07.097619+08:00'
plan: PLAN.md
current: M33
execution_status: profiling_requested_whole_layer_executor_pending
next: M33
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

## M32 — 当前交付

[M32结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)、[两进程九点计时](../../experiments/persistent-tile-remote-20261019/PERFORMANCE.md)。用户最新限定一层完整Transformer block，已取消本轮30层迁移/执行；最终模型E2E后续本机。远端~/cxz隔离CUDA12.4/Torch2.9 cu126编译；baseline与依赖88项不变。

attention/FFN1阶段内实现常驻CTA固定步长tile循环；完整self/cross/FFN与norm/RoPE/调制/残差仍在同一层。QKV/Wo/FFN2仍cuBLASLt，尚无跨阶段ready、warp专职分工或新shared-prefix复用。默认sequential，静态/persistent及FFN1独立开关。本机旧M31 DSO需重编新ABI，远端新DSO已验证。

9真实block（B2/B14 action及B2 Q128 video×三KV）输出逐位；90有效KV连续状态、输入/RNG/参数/API门通过。tile/persistent相对独立顺序路径24中间量，action融合23；48原语检查确认每tile恰好一次与常驻CTA实际多tile执行。无baseline模块/旧DSO加载，原Torch为远端oracle。

两进程378个原始计时样本：persistent+FFN1相对静态+FFN1，B14慢4.42%–6.26%，B2 action慢2.56%–2.80%；未取得新增persistent收益。A/A最大10.07%保留，GPU1/host共享负载及三样本中位限制明确；不是最强B14/Graph或E2E。

五GPU作业77.927秒/731block/48原语/0full30/0episode，全部owned退出。首计时在0block因NVML宿主/容器PID误判退出，修正资源工具后新attempt2/3；传输失败、原收据均保留。CUDA active时间UNKNOWN。原3h墙钟/30min GPU预算收口，不转授下一轮。

## 既有证据与质量线

M31独立静态子图9block/72状态及独立进程54状态是旧证据；M28六actual30/动态block性能是旧版本，不继承到新M32。M21/M25/M26 FFN2/NCU/M27与旧守卫失败保留。完整历史见[归档](history/M32-before-summary-20261019/PROGRESS.md)。

真实D3072/H24/d128/FFN14336；action Q16、Linear M32/224，每attention实例Q16；self-K288/1152/2160，video本批272/1136/2144，cross512。M29文献近邻主张门与M30真实DAG仍有效。

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500；[成功率表](../../experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。描述性覆盖，不解除首两S严格配对FAILED、训练/DEV曝光、组件审计与长尾。

Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

TF64/TF_ON/normalizer/tau/K2B2/bonus/prefix_then_cost与BF16/FP32冻结。PF2=SST完整policy，T50更新/51调用含末尾；reset/cache不计周期。所有臂同用512-token BF16 reset bank，cold另列；actual796与800罚分分列。

## 下一步与边界

M33只考虑单层内部profile与有复用价值的tile/子图；尚未启动新GPU作业。本轮不做30层大算子，最终模型E2E后续本机。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

用户最新澄清：要单层统一persistent执行器内的tile调度。M32只是局部原型；M33实际profile已登记，统一整层尚未实现。
