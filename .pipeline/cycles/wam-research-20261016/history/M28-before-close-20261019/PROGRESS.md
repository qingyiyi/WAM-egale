---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T16:09:23.647800+08:00'
current: M28
execution_status: final_native_precision_pass_timing_resource_partial
next: final_dynamic_block_two_process_timing_and_source_report
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Object 1006/1500，T469/C268/P269；2026-10-19T08:06:44Z两活动/自动后继 | [巡检](../../../投机采样/results/runtime/p0-remote-server-health-cpu1-v2-20261019-r1/LATEST.json) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 原无效监督/PARTIAL保留，获准14补采已完；本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE，旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block；旧66/AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 两份代码整理＋30层/性能 | `in_progress` | 新108检查过/432full30；最大中位回归2.18%，严格2.5%非全过/NVML异常 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与有限设备原语 | `completed` | 30eager/24Graph过；FFN1+GELU六点两进程严格过，B14延迟降2.45–3.23%；完整device FFN2失败 | [全表](../../experiments/operator-profile-device-20261019/RESULT.md) |
| `M27` | 短KV/B14源码与性能口径核查 | `completed` | 两次资源中断PARTIAL/一次等待到期；未改内核；融合B14约3%为更快 | [结果](../../experiments/operator-shortkv-b14-20261019/RESULT.md) |
| `M28` | 完整block原生baseline与optimized | `in_progress` | 最终vector六中间量/六30层/24状态门过；计时受资源打断，待双进程 | [当前](../../experiments/block-native-baseline-20261019/PLAN.md) |

## M28已观察

两份研究入口均为完整native block；自有Norm/RoPE/pointwise/KV/Flash CUDA，GEMM直接cuBLASLt，保留原FP32调制表/Norm参数与BF16计算。Tensor allocator/stream/handle仍用PyTorch，无ATen linear/LayerNorm/SDPA数学转调。optimized仅Q16多FFN1＋GELU；不是单kernel/30层fusion。

最终向量复制版六真实action的23关键量/六原语逐位过；六actual30 query的baseline/optimized最终输出、全部有效KV、metadata、输入/RNG、参数地址/安装还原过；Q128及连续write1→write2→read0共24状态门过。B14 full30为旧真实verifier，B2为同root CFG0/7派生；不称保存的原顺序B2。视频仅block0完整门，不称完整视频30层已验。

旧视频缺字段PARTIAL与首次真实模型FP32参数契约拒绝保留；均后续独立namespace补齐，不修改原FAILED。新FFN2是原Torch配置cuBLASLt通过，旧CUTLASS FFN2失败仍在。

首轮scalar动态block9点跑齐：B14中/长baseline约慢3%；独立profile发现pack/serial slot为主要额外开销。已改为16字节字节复制与CUB稳定physical-order scan，原语/状态/30层重过。最终vector计时被外来PID打断，部分值保留但不签完整双进程；继续自然空闲。未与最强shared-prefix/缓存Graph混比，不声称MegaTransformer或任务E2E收益。

## 质量线与边界

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500；首两S严格配对FAILED、训练/DEV曝光和长尾保留。Object动态数见表，原两卡自动队列/截止不重置。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost及BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾，reset/cache不计周期。所有臂沿用512-token BF16优化reset，cold另列；actual796与800罚分分列。

投机源码/配置/运行依赖与原tile只读；本轮按用户新授权可改算子baseline，旧源已归档。1875投机源与保护state/rules复核不变，不建缺失cycle.yaml。不杀他人/改服务/时钟；BAAI写入仅~/cxz；AutoDL不再SSH、关机UNKNOWN。仅main公开镜像。

不自动训练、120/独立TEST、新TF正式45chunk、第二环境或大型fusion。中文论文仍在会话交飞书。历史和旧数值范围见[归档](history/M28-before-final-native-result-20261019/PROGRESS.md)，当前next只在M28内执行。
