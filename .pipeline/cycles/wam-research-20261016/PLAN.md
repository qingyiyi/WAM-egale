---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T12:07:24.496573+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与算子研究

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

## 当前可见计划

- [x] M25迁入独立`算子研究/wam_ops/baseline`和`optimized`，清晰命名；删除无外部投机调用的16旧算子目录，源码308份存Git。
- [x] CPU核对block forward与原版AST一致、参数/cache原位安装/还原、新CUDA扩展导入；投机1875源码/配置逐字保护。
- [x] 完成两独立进程的六block/六实际30层、三臂数值/有效状态/RNG、write1/2；派生B2来源明示。
- [x] 三轮交错计时及全部原始值已采；三个attempt均有资源冲突，原FAILED/PARTIAL不改。
- [x] 用户空闲继续的新[有界接续](../../experiments/operator-stable-timing-20261019/RESULT.md)完成：432full30/444block、131.338 GPU生命周期秒；108检查逐位通过，自有退出。正式无争用/严格2.5%仍未成立。
- [x] 统一代码讲解、结果/HW与main同步；当前科学边界显式保留。远端既定Object每5分钟巡检，原规则不自动重试。
- [ ] 下一步先讨论完整block分解与第一个优化子图；baseline固定，只在optimized开发，不先大融合。原正式性能缺口保留；T25/T17及Long后两轮仍待登记。

## 范围与预算

M16原10/17 13:55:56起，10/19 13:40:56停新/13:55:56清理，6001000尝试/10GiB；实际1593717，Long已完。M17原10/17 17:11:31起，10/21 16:56:31停新/17:11:31清理，18723000尝试/15GiB；Object新授权14/56000，原无效14的5876已扣，remaining6106124，起点不重置。新匹配T200沿M17总截止，旧T_POOL过期截止未恢复。

M24今晨新GPU原08:48:46起/10:48:46清理，412/5000 block，worker均退出；旧CPU预算和M22 1094.781/1200秒不挪用。M25源自用户新任务，10:14:26起/13:14:26清理，单A100/CPU≤2，GPU累计生命周期≤1200秒，full30≤150、block≤500、存储≤1GiB；等待在全任务3小时内计账，两个独立进程，GPU争用时不开计时。实际累计150 full30/168 block、GPU生命周期137.306秒，worker均自然非零退出（资源中断），调用帽已用满。详细实际数见各Experiment。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/FAILED/paused72/96与MLP0/22缺收据/default A_off不改。TF64/TF_ON、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义不变。PF2=SST完整policy，T50更新/51action调用含末尾。所有臂用512-token BF16优化reset，cold另列；actual796和失败罚分800分列。

投机代码/配置/服务/运行依赖不改。原tile是shared-prefix构建依赖，原位只读保留；保护state/cycle/rules与原tile HW不动。不自动新训练、120/独立TEST、被阻塞的新TF正式45chunk、架构搜索/第二环境/撤回融合。BAAI所有写入永远限~/cxz，不杀他人/改系统/时钟，AutoDL不再SSH。GitHub仅main当前镜像；中文论文会话交飞书，不建初稿。


10/22 go/no-go，10/23冻结，10/28主表，11/01 18:00中文证据包。源码与工作清单见[算子README](../../../算子研究/README.md)。本次重建前四文完整保存于[history](history/M25-before-operator-consolidation-20261019/PLAN.md)。

用户新要求两卡各并行eval，运维交接后四worker实际产出完整局，余下固定分片自动接续；原M17预算/截止不重置、投机源码不改。见[运维接续](../../experiments/remote-eval-parallel-20261019/RESULT.md)。
