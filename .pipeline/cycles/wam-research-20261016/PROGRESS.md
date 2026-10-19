---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-19T10:47:48.521957+08:00'
current: M25
next: stable_uncontended_operator_timing_pending_new_bounded_job
execution_status: operator_code_and_full30_parity_complete_timing_resource_pending_GPU_budget_stopped
---
# 当前研究进度

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；严格首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Goal/Spatial各1500局完成；Object14补采/准入/ON完成，匹配T200及全500双卡运行 | [证据](../../experiments/libero-four-suite-morning-20261019/RESULT.md) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 保留原无效监督/PARTIAL；获准14补采已完成，本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE；旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能尾部/异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装，登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block，六点中位比0.9677–1.0033，单轮波动保留；原66/旧AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 算子两份实现整理＋整模型/性能 | `in_progress` | 独立baseline/optimized，CPU绑定/native过；16旧目录已删、308源留Git；投机1875源不变；六点block/实际full30两进程逐位过；计时有资源冲突，150次full30已用满、自有退出 | [证据](../../experiments/operator-consolidation-20261019/PLAN.md) |

## 当前结果与缺口

新算子入口为`算子研究`，仅baseline/optimized两份主实现，公用安装和验证/计时工具。baseline原数学顺序，optimized目前只换独立CUDA SDPA，未实现新MegaTransformer调度。16旧目录已删、308源留Git a0ed2d8；原tile/shared-prefix依赖保留，1875投机源/配置逐字未改。

三个attempt共130项数值/有效KV/输入/RNG检查过、max_abs=0；其中两进程完成六block/六实际full30点、write1/2和安装还原。B14历史gold过，B2 full30明确为根CFG派生诊断，非原始顺序B2。全部计时已保存，但三次结束/阶段资源检查均遇外来GPU进程，原FAILED/PARTIAL保留；波动0.6197–1.2380，不签稳定性能。累计150 full30/168 block、GPU生命周期137.306秒，用满forward帽，自有worker退出、无继续GPU自动启动；原13:14截止不重置。[最终实际结果](../../experiments/operator-consolidation-20261019/RESULT.md)。

M24旧六真实action block0（B2/B14×短中长K）18次GPU数值/有效KV/输入/RNG及历史gold逐位过；294block约15秒退出，六点中位耗时比0.9677–1.0033、单轮0.8587–1.0973。原B0两cache66次精度/旧AA2.512635%超2.5%失败保留，旧watch/fixed3 audit不再跑。原CUDA stride/alias/RNG缺失、video/完整模型未由M24新增证明；新M25独立验证不能回写旧门。

Long T483/C484/P473，800罚分均值285.858/282.588/293.270；共同成功C/T长度max2.329167、P/T3.223577。Goal T463/C464/P467，罚分158.806/157.166/156.466；Spatial T472/C466/P464，罚分145.998/152.964/155.304，各臂500。严格首两S失败：Long C36/P66，Goal C165/P334，Spatial C155/P312；训练/DEV曝光/独立性缺口不改，正式E2E UNKNOWN。

Object旧16有效、旧14监督冲突无效保留；新14完整补采/准入/ON3000完成，匹配T200与全500双卡接续；新health-v2跟踪。额外GPU并发因NVML/容器PID对应未证明继续拦截，不伪称自有。以最新实际巡检为准，不把旧状态当实时。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/FAILED/paused72/96与MLP0/22缺收据/default A_off不改。TF64/TF_ON、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义不变。PF2=SST完整policy，T50更新/51action调用含末尾。所有臂用512-token BF16优化reset，cold另列；actual796和失败罚分800分列。

投机代码/配置/服务/运行依赖不改。原tile是shared-prefix构建依赖，原位只读保留；保护state/cycle/rules与原tile HW不动。不自动新训练、120/独立TEST、被阻塞的新TF正式45chunk、架构搜索/第二环境/撤回融合。BAAI所有写入永远限~/cxz，不杀他人/改系统/时钟，AutoDL不再SSH。GitHub仅main当前镜像；中文论文会话交飞书，不建初稿。

预算见PLAN和登记原件。历史四文完整见[本次归档](history/M25-before-operator-consolidation-20261019/PROGRESS.md)。
