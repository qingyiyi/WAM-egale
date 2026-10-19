---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T19:27:26.713126+08:00'
current: M31
execution_status: complete_isolated_static_tiles_finite_block_validation
next: M32
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

## M31当前交付

[结果](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)、[源码说明](../../../算子研究/wam_ops/optimized/SUBGRAPHS.md)。用户批准隔离执行路径与实现tile子图，已完成独立optimized参数/native/C++与DSO。baseline与依赖88项不变；独立进程不导入baseline或加载旧WAM库。默认sequential保持等价，tiled与FFN1融合独立开关。

真实self/cross/FFN三段按同stream执行；attention显式query tile保持Flash在线softmax/split/combine，FFN1显式row/column MMA+epilogue；QKV/Wo/FFN2仍cuBLASLt。没有persistent队列/跨阶段ready或新调度收益。

9真实block：action B2/B14×短中长、video B2 Q128×短中长。原Torch/baseline/三种optimized输出逐位；seq/tiled每组24中间量，action fused23。72连续KV状态（含驱逐）、保存identity、输入/CPU-CUDA RNG/参数地址与grad flags/安装还原通过；实际kernel名称核验。另一独立进程9输入/54状态通过。

首attempt仅kernel名识别误判REJECTED保留；video旧标签错误按原input_path追溯，最终identity和状态后不变检查补齐。CPU收据finalizer错误保留。3 GPU作业81.007秒、483独立block/0full30/0episode，全部owned退出。新30层因可用约4.9GiB不足28GiB BLOCKED；新性能未测。有限通过不外推整模型/机器人。

## 既有证据

[M28](../../experiments/block-native-baseline-20261019/RESULT.md)旧实现六actual30有限逐位、两进程9点/504样本动态block中位不慢；A/A最大8.63%保留，不据小差值宣称稳定收益。不是最强B14/Graph、完整chunk/E2E。M21/M25/M26 FFN2/NCU/M27旧失败保留。

[M30设计](../../experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)：D3072/H24/d128/FFN14336，action Q16，Linear M32/224；attention每候选仍Q16。action有效K288/1152/2160；video本批272/1136/2144，cross512。MegaMoE只借思想，SM100专有原语不移植到SM80。[M29](../../experiments/speculative-denoising-novelty-20261019/RESULT.md)直接先例与主张门仍有效。

## 质量线

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500；[冻结表](../../experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。描述性覆盖，首两S严格配对FAILED、训练/DEV曝光、组件审计与长尾保留。

Object 1340/1500，完成数T500/C420/P420，来源2026-10-19T11:26:39Z；controller身份匹配=True，原同卡队列继续。 动态源为原CPU只读watch LATEST，约5分钟更新，完成数不是成功数。

TF64/TF_ON/normalizer/tau/K2B2/bonus/prefix_then_cost与BF16/FP32冻结。PF2=SST完整policy，T50更新/51调用含末尾，reset/cache不计周期；各臂同用512-token BF16 reset bank，cold另列，actual796/800罚分分列。

## 下一步与保护

M32尚未启动：资源允许时补新full30/全部有效状态，再按强B14边界考虑有界persistent/ready；主要机制指标ΣT_round/Σg，不用候选数或调用减少代替。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。


本轮前完整进度见[归档](history/M31-before-summary-20261019/PROGRESS.md)。旧事实/失败不被摘要改写。
