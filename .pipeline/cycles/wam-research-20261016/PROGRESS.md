---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-20T00:26:37.588785+08:00'
plan: PLAN.md
current: M35
execution_status: engineering_timing_and_operator_CPU_parallel
next: M34
---
# 当前研究进度

## M17 — 四套既定评估：计算完整、审计未放行

四套共6000完整episode，每套T/C/P各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。[逐局CPU核对及长度/罚分](../../experiments/libero-four-suite-performance-20261019/RESULT.md)。6000不是独立TEST；首两S严格配对FAILED、训练/DEV曝光、组件审计和长尾保留。原算法失败不删除，actual796与800罚分分列。既定GPU计算已清理。

## M31 — 独立静态tile子图：completed

[原证据](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)：9block/72状态及独立进程54状态有限通过；不继承给新算术。baseline及依赖冻结，优化只在optimized。

## M32 — 阶段内persistent：completed

[原结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)：9真实block/90有效KV/48原语通过；两进程378样本，新增persistent比静态B14慢4.42%–6.26%，B2 action慢2.56%–2.80%，A/A最大10.07%。只在attention/FFN1阶段内循环，QKV/Wo/FFN2仍host cuBLASLt，不满足整层单kernel。原五GPU作业77.927秒/731block，0full30/episode、owned退出。

## M33 — 退化profile：completed

[实测归因](../../experiments/single-block-persistent-profile-20261019/RESULT.md)：B14长KV两模式均34 kernel；退化207.072us由self/cross/FFN1三项207.2425us解释。216 CTA反事实使短中长退化4.154/4.781/5.459%降至0.726/0.474/0.179%，仍无加速。NCU权限失败，实测occupancy/L2/DRAM UNKNOWN；两次扩展身份实施失败保留。5 GPU jobs81.856秒/293尝试，0full30/episode，22:53清理。baseline88项不变。

## M34 — 整层单persistent kernel、warp分工：in_progress

最新用户要求现在直接实现，明早交实际速度/精度/问题。整层[source](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[Python入口](../../../算子研究/wam_ops/optimized/whole_layer.py)、[说明](../../../算子研究/wam_ops/optimized/WHOLE_LAYER.md)已创建，独立DSO已CPU编译成功。self/cross、全部投影、FFN、Norm/RoPE/调制/残差/KV管理都在cooperative kernel；8 warp中调度/两加载/四计算/写回分工，阶段grid barrier保证DAG，未声称跨阶段细流水。

编译1参数遗漏、编译2私有工具链link.stub缺失均保留；后续向量加载与真实4D RoPE接口修正版本已编译。CUDA源码df8b386...。本机自然空闲先验证：[实际9输入](../../experiments/single-layer-warp-specialized-20261019/LOCAL-CORRECTNESS.md)，本机2条、远端9条trace各1整层kernel，tile/输入/RNG/参数门通过；9输出逐位FAIL、relative L2 0.00131–0.00560，只读9KV通过、写序列18中9KV数值失败，metadata均过。首本机误用默认Python依赖失败0block，已退出，原记录保留。原cuBLASLt API重建得到B2 FFN2 splitK7、B14 Q/FFN2 splitK2，inplace；不是已结束调用直接拦截，不能单凭配置定根因。

实际函数172 registers/thread、59616B shared、LOCAL0、理论1 CTA/SM/108 CTA；不是实测achieved occupancy。未通过原数值门，生产不接入，四false不变。首批远端3作业已关闭：450 block/270计时样本，新kernel比native慢5.12–11.28倍，全部owned退出。另登记83调用/15min阶段计时与至多一个ldm40诊断，CPU已编译profile，等Object计时关闭；不改主源码/DSO，最晚08:30清理。[登记](../../experiments/single-layer-warp-specialized-20261019/PLAN.md)。

## M35 — 四套新独占工程计时：in_progress

[新登记](../../experiments/libero-four-suite-performance-20261019/ISOLATED-MEASUREMENT-PLAN.md)：每套10task/init49×三臂×3独立server进程重复，最多360完整局。GPU0 Goal→Object；GPU1 Spatial→Long；每卡单target worker、server/client各2核。新重复只作计时，不增加独立质量N。Goal/Spatial各90计时局齐；Long已自动READY执行，Object仅原未启动repeats2/3接续。Object repeat1误用Goal/Spatial迁移入口被白名单拒绝，0局/0teacher/原guard CLEAN；失败保留不重跑，原360预计最多330，面板PARTIAL。新continuation不改原截止/预算，未改投机源码或旧白名单。；队列实时数以远端原终态为准。[部分新计时表](../../experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)。

新预算6h/卡、1452000尝试、3GiB；05:34停止新工作、05:44清理（10/20北京，以原REGISTRATION秒为准），不延期。三小套复用冻结入口；Long用独立10case资源schema，不伪造原500PASS。争用标污染、完整episode边界停，不杀他人、不自动重试。旧受阻45chunk未启动，四false不变；这是用户新授权的工程计时而非旧质量证书放行。CPU编译不与GPU计时混跑同卡。

## M36 — RoboTwin迁移只读调研：completed

[结果](../../experiments/robotwin-migration-survey-20261019/RESULT.md)：官方LingBot-VA RoboTwin权重存在，50唯一任务；单条件100有效scene/task，50×100×三臂=15000局。旧官方client先专家筛seed，不是LIBERO50固定init。动作16有效通道/30padding、Q32，不能直接用当前7通道/Q16 draft。固定代码/配置/权重metadata身份，未下载模型或装环境、未启第二环境。

## 当前接口与保护

D3072/H24/d128/FFN14336；action Q16/Linear M32/224，每attention实例Q16；self K288/1152/2160，video272/1136/2144，cross512。候选private KV独立。TF64/TF_ON、normalizer/tau/K2B2/bonus/prefix_then_cost/BF16-FP32冻结；PF2=SST完整policy，T50更新/51action调用含末尾。双方同用512-token BF16 reset bank，cold另列。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。M21/M25/M26/M27旧失败与负结果保留，原tile/投机代码与运行依赖不动。

BAAI所有写入限~/cxz；AutoDL已归还、不再SSH。最新允许性能统一远端，替代最终E2E必须本机。仅一层、baseline/optimized两主实现；baseline含共享依赖冻结，88项CPU核对未变。state/cycle/rules保护，不旧sync/升级HW、不建缺失cycle.yaml。无自动新训练/120/独立TEST/第二环境/30层大融合；只main公开核对源码/HW/小证据，模型/PT/KV/DSO/凭据不传。

M17原预算另存：10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB，不能借M35重置。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。完整前状态见[归档](history/M34-start-20261020/PROGRESS.md)。
