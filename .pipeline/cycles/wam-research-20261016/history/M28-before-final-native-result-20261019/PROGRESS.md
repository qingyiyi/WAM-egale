---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T14:49:18.383103+08:00'
current: M28
execution_status: final_native_precision_pass_timing_resource_partial
next: complete_final_two_process_dynamic_block_timing_at_natural_idle
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Goal/Spatial全齐；06:46:38UTC Object880/1500，T426/C227/P227；3worker/自动排队 | [巡检](../../experiments/operator-shortkv-b14-20261019/REMOTE-STATUS-READONLY.json) |
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
| `M28` | 完整block原生baseline与optimized | `in_progress` | 完整native action六点/optimized六点过；Q128/30层/性能待门 | [计划](../../experiments/block-native-baseline-20261019/PLAN.md) |

## 当前算子结论

M28当前baseline/optimized研究入口已接完整native block：自有Norm/RoPE/pointwise/KV/Flash CUDA，GEMM复用公开的cuBLASLt。baseline六action每点23关键量/六原语逐位过，optimized六action亦过；12合法连续写/读状态门过，含long驱逐。已补3份真实Q128并过24状态门；最终vector版六中间量/六30层及安装还原均过。完整动态计时受外来PID中断，旧scalar9点与新vector部分值仅各自有限证据，最终双进程待齐；投机生产不切换。[当前计划](../../experiments/block-native-baseline-20261019/PLAN.md)。

以下M25–M27是旧attention替换/可选FFN1路径的历史结果，不自动继承到M28完整native。

M26六真实block的30eager/24Graph过；FFN1+GELU六点两进程严格过，B2延迟降0.74–1.52%、B14降2.45–3.23%。完整device FFN2失败、四个分块公式未复现，根因UNKNOWN。原17作业751.141秒、NCU默认boost漏关偏差和旧NVML缺口保留。

M27确认用户关注自有实现/Torch；上次B14约3%是融合加速而非退化。未融合M25 B14 block中位耗时变化−1.82%～+1.19%，不代表旧54异常消失。新同路径cached Graph短/中B14比0.99897/1.00139；两次PARTIAL/一次无context等待到期，无完整新六点矩阵。固定Torch和自有kernel tile/split一致；255regs/64KiB元数据同，compiler/SASS贡献未知。665.685生命周期秒含等待，GPU阶段墙钟31.145秒，CUDA active UNKNOWN；6180attention/918block intents、0full30/新episode，三个owned全部退出。未改数学内核。[分表与原始证据](../../experiments/operator-shortkv-b14-20261019/RESULT.md)。

## 质量线与约束

Long483/484/473、Goal463/464/467、Spatial472/466/464，各臂500；首两S配对审计FAILED/训练DEV曝光/成功长尾保留。Object最新只读巡检见表，3worker活动/后继自动，原截止不重置。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期。各臂512-token BF16优化reset，cold另列；actual796/800罚分分列。

投机源码/配置/运行依赖与原tile只读；M28按用户新授权可改算子baseline，旧源已归档。1875投机文件及保护state/rules不改，不建缺失cycle.yaml。不自动训练、120/独立TEST、新TF正式45chunk或第二环境。BAAI写入仅~/cxz，不杀他人/改服务/时钟；AutoDL不再SSH，关机UNKNOWN。仅main公开镜像；中文论文会话交飞书，不建初稿。

更早完整进度见[本次历史](history/M27-before-investigation-20261019/PROGRESS.md)，仅按next读取原证据。
