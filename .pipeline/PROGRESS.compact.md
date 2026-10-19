# 当前短入口

更新：2026-10-20T00:26:37.588785+08:00。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先约束记忆，再读当前四文；旧大索引/原trace按需。

## 当前：M35远端工程计时运行；M34并行整层warp专职kernel编码

四套原6000局CPU复核完成，每臂500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499（T/C/P）。用户新允许统一远端测性能，替代最终E2E必须本机。新[独占工程计时](experiments/libero-four-suite-performance-20261019/ISOLATED-MEASUREMENT-PLAN.md)已启动GPU0 Goal、GPU1 Spatial，Object/Long接续；10task/init49×三臂×3进程重复，最多360完整局，6h/卡、1452000尝试、3GiB，现场绝对截止不延期。每卡一个worker；旧45chunk入口不恢复，旧四false/FAILED不变。新重复不增加独立质量N。

最新用户要求立即实施。M34整层单kernel源码和独立DSO已CPU编译；调度/两加载/四计算/写回warp分工，cooperative阶段DAG，本机9真实输入已重放，2 trace证实单launch、tile/输入/RNG过，但9输出逐位FAIL，relative L2 0.13%–0.56%；9只读KV过，写序列18中9数值失败。实际172寄存器/thread、理论1CTA/SM、LOCAL0；不是运行occupancy，不集成。远端9 trace也单kernel、原数值FAIL；450 block/270时间样本，比native慢5.12–11.28倍，首批已清理。另有83调用/15min的profile/最多一个布局诊断，等Object计时关闭，不改主源码/DSO，最晚08:30清理。baseline88项未变。见[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。M36[只读调研已完成](experiments/robotwin-migration-survey-20261019/RESULT.md)，官方50任务/权重存在，Q32/16通道需新draft接口，不启动第二环境。

## M33已完成与保留的负结果

[M33实际profile](experiments/single-block-persistent-profile-20261019/RESULT.md)完成。B14长KV静态/阶段内persistent均34 kernel，增量207.072us由self/cross attention与FFN1合计207.2425us解释。108 CTA上限低于理论2 CTA/SM容量；隔离216 CTA反事实使B14短中长相对静态退化从4.154%/4.781%/5.459%降到0.726%/0.474%/0.179%，没有新增加速。不锁时钟，跨case绝对时间非单调，不能作正式性能。

九真实block×四模式两次profile（153调用）、有效三B14反事实112调用，输出/有效KV/RNG通过。NCU权限失败，实测occupancy/L2/DRAM UNKNOWN；两个扩展身份/符号实施失败保留，第三attempt隔离后完成。五GPU job生命周期81.856秒、293 block尝试含失败、0full30/0episode；22:53两GPU无进程，全部owned退出，新证据42.91MiB。baseline及依赖88项不变。

用户要求的是**整个单层统一persistent kernel，内部warp组调度全部tile**。当前M32只有attention/FFN1阶段内常驻循环，QKV/Wo/FFN2仍host cuBLASLt，不满足整层目标。[完整DAG与缺口](experiments/single-block-persistent-profile-20261019/WHOLE-LAYER-EXECUTOR-GAPS.md)已登记，M34新源码本机结构验证完成，原精度门FAIL，远端时间待验。下一M34验证设备原语、统一执行器和状态，不扩30层，不以新wrapper或多kernel包装宣称完成。

M32九真实block/90有效KV/48原语有限逐位与两进程378样本原结果保留；B14慢4.42%–6.26%、A/A最大10.07%，不能用本轮诊断覆盖。

## 历史与机制边界

[M31](experiments/transformer-tile-subgraph-20261019/RESULT.md)独立静态路径9block/72状态及独立进程54状态；[M28](experiments/block-native-baseline-20261019/RESULT.md)六actual30及两进程动态block是旧版本结果，不自动套给新M32。M21/M25严格2.5%/NVML、M26 CUTLASS FFN2/NCU、M27资源与旧误判均保留。

[M30真实DAG](experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)：D3072/H24/d128/FFN14336，action Q16/Linear M32或224，每attention实例仍Q16；action self-K288/1152/2160、video本批272/1136/2144、cross512。候选private KV不共享。MegaMoE只借思想，SM100原语不照搬SM80。

[M29文献与主张门](experiments/speculative-denoising-novelty-20261019/RESULT.md)：ICML2025投机扩散、ICLR2026 FlowCast、TS-DP/FLASH、Speculative Forcing/WAMJET有直接近邻。复现/普通batch/缓存/现有融合不能单列新贡献。

## 远端质量与资源

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500完整；[成功率表](experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

原CPU约5分钟巡检继续；动态数看原watch的LATEST，SSH失败不冒充实时。首两S严格配对FAILED、训练/DEV曝光、组件审计与长尾仍有效；描述性覆盖不是独立TEST或非劣。

TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost及BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期，不是bonus或状态恢复。所有臂同用512-token BF16 reset bank，cold另列；actual796/罚分800分列。

## 下一步

M34：整层统一persistent kernel是用户明确目标。设备原语→整层ready/warp组执行→完整输出/KV/RNG→强baseline性能。M33已关闭；M35独占工程计时运行，M34本机有限正确性已收口、远端CPU待机等GPU0释放。下一作业需有界登记，M33余额不转授；不扩30层。

CUDA13.x：本机driver595.71.05，远端535.161.08；远端不满足普通CUDA13最低580条件。未升级驱动或系统，本轮独立CUDA12.4/cu126，未来另核验。

## 保护

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

约束入口：[memory/INDEX.md](memory/INDEX.md)。旧方案以Cycle归档追溯，不重复全量展开。

M35补充：Object首轮误送旧Goal/Spatial入口，0局0teacher，保留且不重试；只接续原未启动2/3，原360最多330，面板PARTIAL。Goal/Spatial各90计时局齐、Long自动运行；同卡单worker/原05:34停新与05:44清理不变。
