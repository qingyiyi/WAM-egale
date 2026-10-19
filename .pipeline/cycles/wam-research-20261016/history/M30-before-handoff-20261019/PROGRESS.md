---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T17:23:28.879867+08:00'
current: M30
execution_status: CPU_shape_DAG_and_tile_persistent_handoff
next: verify_actual_shapes_and_build_dependency_outline
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Object 1075/1500，T491/C291/P293；2026-10-19T08:51:38Z两活动GPU1，GPU0分片已完/后继排队 | [核对](../../experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 原无效监督/PARTIAL保留，获准14补采已完；本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE，旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block；旧66/AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 两份代码整理＋30层/性能 | `superseded_with_open_findings` | 新108检查过/432full30；最大中位回归2.18%，严格2.5%非全过/NVML异常 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与有限设备原语 | `completed` | 30eager/24Graph过；FFN1+GELU六点两进程严格过，B14延迟降2.45–3.23%；完整device FFN2失败 | [全表](../../experiments/operator-profile-device-20261019/RESULT.md) |
| `M27` | 短KV/B14源码与性能口径核查 | `completed` | 两次资源中断PARTIAL/一次等待到期；未改内核；融合B14约3%为更快 | [结果](../../experiments/operator-shortkv-b14-20261019/RESULT.md) |
| `M28` | 完整block原生baseline与optimized | `completed` | 六actual30/24状态/原grad flags逐位；两进程九点齐，非最强B14/Graph或E2E | [结果](../../experiments/block-native-baseline-20261019/RESULT.md) |
| `M29` | 投机去噪文献重合与远端核验 | `completed` | TS-DP/FlowCast/FLASH直接先例；新增调度仍是假设 | [文献与主张门](../../experiments/speculative-denoising-novelty-20261019/RESULT.md) |

| `M30` | tile/persistent方向与协作形状交付 | `in_progress` | CPU-only核验真实shape/参数/DAG，不先预设tile收益 | [登记](../../experiments/transformer-tile-handoff-20261019/REGISTRATION.json) |

## M28已完成

baseline/optimized均是完整native顺序block；Norm/RoPE/pointwise/KV/Flash CUDA、直接cuBLASLt，保留原FP32参数/BF16舍入。Tensor allocator/stream/handle仍依赖PyTorch。optimized仅Q16多FFN1＋GELU；不是单kernel/30层fusion。

最终六action的23关键量/六原语、24action/video写读状态、12optimized状态门逐位过。六actual30 query的baseline/optimized输出、全部有效KV/metadata/input/RNG/参数地址/安装还原通过。原839个requires_grad=True标志保留的最终18模型调用亦过。B14为旧真实verifier，B2为root CFG0/7派生；视频只验证block0。

最终vector两独立进程各9点/504原值齐，动态block相对两个原Torch中位均不慢；A/A最大8.63%偏差保留。不与最强shared-prefix/缓存Graph混比；Graph尚不支持，没有本轮新30层/chunk/episode速度。旧缺字段/混合dtype拒绝/资源PARTIAL、M25严格门/NCU与FFN2旧失败均保留。23GPU作业1088.541/3600生命周期秒、288模型/5114独立block/144原语，owned全退出。[全结果与源码](../../experiments/block-native-baseline-20261019/RESULT.md)。

## M29文献判断

ICML2025投机扩散、ICLR2026 FlowCast、TS-DP单层Transformer蒸馏/机器人batch verify、FLASH动作chunk/PF已有直接先例；Sequoia/Cascade已有树成本/共享prefix。新近Speculative Forcing用于视频世界模型，WAMJET已包含LingBot-VA常规Graph/融合。当前复现和普通copy/FFN优化不足以证明创新。[原文清单与可写主张](../../experiments/speculative-denoising-novelty-20261019/RESULT.md)。潜在增量是候选感知执行超越最佳B14并减少sum(T_round)/sum(g)，目前仍待验证。

## 质量线与边界

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500；首两S严格配对FAILED、训练/DEV曝光和长尾保留。Object动态数见表，原两卡自动队列/截止不重置。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost及BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾，reset/cache不计周期。所有臂沿用512-token BF16优化reset，cold另列；actual796与800罚分分列。

投机源码/配置/运行依赖与原tile只读；本轮按用户新授权可改算子baseline，旧源已归档。1875投机源与保护state/rules复核不变，不建缺失cycle.yaml。不杀他人/改服务/时钟；BAAI写入仅~/cxz；AutoDL不再SSH、关机UNKNOWN。仅main公开镜像。

不自动训练、120/独立TEST、新TF正式45chunk、第二环境或大型fusion。中文论文仍在会话交飞书。历史和旧数值范围见[归档](history/M28-before-final-native-result-20261019/PROGRESS.md)，当前next为M30 CPU形状/DAG与tile-persistent交付；后续GPU实验另有界登记。
