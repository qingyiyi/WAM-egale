---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-19T09:09:48.963521+08:00'
current: M24
next: wait_clean_local_timing_window_and_supervise_approved_Object_recovery
execution_status: finite_GPU_B0_accuracy_complete_AA_failed_remote_Object_recovery_running
---
# 当前研究进度

本机完整block GPU精度已66次通过，性能因A/A和NVML异常待定；BAAI Goal/Spatial各1500局完成，Object14补采与匹配T200双卡真实运行。宏观completed不表示质量门通过。

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | 描述性完成，T483/C484/P473；配对FAILED/曝光/四false保留 | [完整结果](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial固定链 | `in_progress` | Goal/Spatial各1500局完成；Object固定14补采与匹配Target200已双卡接续，后续ON和全500自动排队 | [晨间结果/修复](../../experiments/libero-four-suite-morning-20261019/RESULT.md) |
| `M18` | 迁移原件/环境、释放旧机 | `completed` | 四套资产落BAAI；AutoDL不再SSH、关机UNKNOWN | [迁移](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限证据，不外推 | [576检查/60门/420轮](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评测迁新机 | `completed` | 原Object16有效，原新14完整但监督无效，今晨获准固定14补采；本机不自动接质量 | [资源接续](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE；旧失败/PARTIAL保留 | [旧报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | Torch对齐与有限性能 | `completed` | 108attention/54block/六full30/三policy严格数值过；旧计时有尾部与资源异常 | [结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；不再迁移开发，用户kill模拟器后已退出 | [安装结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 完整block显式顺序B0 | `in_progress` | CPU最终复核/22契约过；今晨六真实fixture×cache的66次GPU逐位/有效状态过；A/A超2.5%停止，性能PENDING | [GPU接续](../../experiments/wam-block-gpu-followup-20261019/RESULT.md) |

## 结果与边界

Goal T463/C464/P467，各500；Spatial T472/C466/P464，各500。800罚分均值Goal158.806/157.166/156.466、Spatial145.998/152.964/155.304。两套均有FAILED_FIRST_TWO_S_PAIRING，训练曝光/独立性与正式E2E未获解决；不把1500完整终态写成质量PASS。Long原结果和长尾见原汇总，不重算旧门。

B0是新Python阶段组织、Torch原基元；GPU六保存真实action block0、两种cache、66次逐位/全部有效KV/元数据/输入/当前RNG与旧GPU gold通过。原CUDA stride/alias/RNG未持久化，视频完整block/B0完整30层/policy未由本轮证明。A/A最大2.512635%>2.5%，停止且不放宽；本机0显存/无进程却NVML24–25%，根因UNKNOWN，不能签性能。CPU监控同一10:48:46截止。

Object旧16有效，新14完整机器人终态但监督文件冲突无效；旧失败保留，用户今晨明确批准仅补14。v3首尾/物理卡/训练角色不变，T200先作为独立baseline并行，未用来重训ON。自动链/5分钟巡检见新health-v2，不声称待运行阶段已启动。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/paused72/default不改。原tile与保护源不动，不自动恢复120/独立TEST/新TF45chunk/撤回融合/架构搜索。详细预算/原截止见PLAN和各Experiment。

此前四文完整保留：[本次history](history/M24-before-GPU-followup-20261019/PROGRESS.md)。
