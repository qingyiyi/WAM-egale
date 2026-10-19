---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T13:37:18.769422+08:00'
current: M26
execution_status: complete_bounded_steps_partial_device_support
next: discuss_complete_block_DAG_and_FFN2_boundary_after_result_handoff
---
# 当前研究进度

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Goal/Spatial全齐；13:31 Object668/1500，T356/C156/P156；3worker活动，后继自动排队 | [巡检](../../experiments/operator-profile-device-20261019/REMOTE-STATUS-READONLY.json) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 原无效监督/PARTIAL保留，获准14补采已完；本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE，旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block；旧66/AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 两份代码整理＋30层/性能 | `in_progress` | 新108检查过/432full30；最大中位回归2.18%，严格2.5%非全过/NVML异常 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与有限设备原语 | `completed` | 30eager/24Graph过；FFN1+GELU六点两进程严格过，B14延迟降2.45–3.23%；完整device FFN2失败 | [全表](../../experiments/operator-profile-device-20261019/RESULT.md) |

## 本轮算子结论

强基线30eager/24Graph配置全部逐位/状态过，真block0 M32→224、每候选Q16；强cache profile的FFN占kernel时长21.9–29.9%。完整device FFN2六点失败，四个分块公式均未复现，根因UNKNOWN。最终FFN1+GELU保留中间BF16舍入、FFN2保持Torch，六点×两独立进程中间量/完整block/有效KV/metadata/输入/RNG/参数字节过；B2延迟下降0.74–1.52%、B14 2.45–3.23%，仅探索性block，不是MegaTransformer调度/30层/chunk/E2E结论。

本轮1093 block host/capture/replay intents、617原语、751.141生命周期秒，17作业终态、两工具失败保留、无timeout，自有GPU退出。NCU初轮漏显式关闭默认boost，实际改频UNKNOWN；已记偏差、后续入口强制none，全部counter仅带此边界诊断。NVML空卡24%异常和M25严格±2.5%未全过不改。[全结果](../../experiments/operator-profile-device-20261019/RESULT.md)、[源码](../../../算子研究/README.md)。

## 质量线与运行

Long T483/C484/P473，各500；罚分均值285.858/282.588/293.270，成功长度长尾保留。Goal463/464/467、Spatial472/466/464，各500；原首两S审计FAILED/数据曝光不改。Object13:31为668/1500，T356/C156/P156，3活动/其余自动排队、无terminal；每5分钟watch继续，原M17截止不重置。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期。各臂同用512-token BF16优化reset，cold另列；actual796与800罚分分列。

投机源码/配置/运行依赖和baseline不改；本轮1875投机文件复核不变。原tile/shared-prefix依赖原位只读，保护state/cycle/rules不覆盖，不建缺失cycle.yaml。不自动新训练、120/独立TEST、新TF正式45chunk或第二环境。BAAI写入永远仅~/cxz，不杀他人/改服务/手动时钟。AutoDL不再SSH，关机UNKNOWN。GitHub仅main镜像；中文论文会话交飞书，不建初稿。

完整旧进度保存在[本次历史](history/M26-before-result-20261019/PROGRESS.md)；最新证据按本表next按需读取，不展开旧大索引。
