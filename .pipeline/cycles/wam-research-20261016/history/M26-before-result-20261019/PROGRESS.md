---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-19T12:51:06.762666+08:00'
current: M26
next: run_bounded_strong_baseline_profile_then_device_primitive
execution_status: M26_registered_CPU_preparation
---
# 当前研究进度

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；严格首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Goal/Spatial全齐；Object匹配T200已完，四worker/两卡各2已实际完整出局，剩余分片自动接续 | [并发证据](../../experiments/remote-eval-parallel-20261019/RESULT.md) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 保留原无效监督/PARTIAL；获准14补采已完成，本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE；旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能尾部/异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装，登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block，六点中位比0.9677–1.0033，单轮波动保留；原66/旧AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 算子两份实现整理＋整模型/性能 | `in_progress` | 新接续两进程108项逐位过，432full30/444block；无外来PID，完整模型中位回归≤2.18%；NVML异常/严格2.5%仍未全过 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与设备原语1/2 | `in_progress` | 用户已同意；单block真实输入，先profile选热点；不做scheduler，不改投机 | [登记](../../experiments/operator-profile-device-20261019/PLAN.md) |

## 当前结果与缺口

新算子入口为`算子研究`，仅baseline/optimized两份主实现，公用安装和验证/计时工具。baseline原数学顺序，optimized目前只换独立CUDA SDPA，未实现新MegaTransformer调度。16旧目录已删、308源留Git a0ed2d8；原tile/shared-prefix依赖保留，1875投机源/配置逐字未改。

原M25三attempt130检查过、150full30/168block，但资源冲突FAILED/PARTIAL保留。用户新要求空闲继续后，另登记有限接续：两进程108检查逐位过、432full30/444block、131.338 GPU生命周期秒、自有退出；102次采样无外来PID。完整模型baseline/原版0.9539–1.0058、optimized0.9535–1.0218；±2.5%非全点通过，零显存时NVML24–25%原因UNKNOWN，不签正式无争用。B14真实full30，B2 root CFG派生边界不改。[新结果](../../experiments/operator-stable-timing-20261019/RESULT.md)，下一步讨论block分解与单子图，baseline固定。

M24六fixture/18次逐位、294block及0.9677–1.0033历史耗时保留；原66 GPU检查/旧AA2.512635%超2.5%失败不改，新M25不回写旧门。原CUDA布局/别名/RNG未保存、video block缺fixture的边界继续存在。

Long T483/C484/P473，800罚分均值285.858/282.588/293.270；共同成功C/T长度max2.329167、P/T3.223577。Goal T463/C464/P467，罚分158.806/157.166/156.466；Spatial T472/C466/P464，罚分145.998/152.964/155.304，各臂500。严格首两S失败：Long C36/P66，Goal C165/P334，Spatial C155/P312；训练/DEV曝光/独立性缺口不改，正式E2E UNKNOWN。

Object14补采/准入/ON均完成，匹配T200完整。用户要求两卡各并行，唯一上下文集合/新增delta映射后只交接运维CPU dispatcher；四worker实际完整出局，模型/client/guard无信号，投机算法/评估源码不改。12:05为去重435/1500，T278/C78/P79、77组三臂齐；余下0_99/100_199自动排队，每5分钟watch继续，原M17截止/预算不重置。[并发结果](../../experiments/remote-eval-parallel-20261019/RESULT.md)。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/FAILED/paused72/96与MLP0/22缺收据/default A_off不改。TF64/TF_ON、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义不变。PF2=SST完整policy，T50更新/51action调用含末尾。所有臂用512-token BF16优化reset，cold另列；actual796和失败罚分800分列。

投机代码/配置/服务/运行依赖不改。原tile是shared-prefix构建依赖，原位只读保留；保护state/cycle/rules与原tile HW不动。不自动新训练、120/独立TEST、被阻塞的新TF正式45chunk、架构搜索/第二环境/撤回融合。BAAI所有写入永远限~/cxz，不杀他人/改系统/时钟，AutoDL不再SSH。GitHub仅main当前镜像；中文论文会话交飞书，不建初稿。

预算见PLAN和登记原件。历史四文完整见[本次归档](history/M25-before-operator-consolidation-20261019/PROGRESS.md)。
