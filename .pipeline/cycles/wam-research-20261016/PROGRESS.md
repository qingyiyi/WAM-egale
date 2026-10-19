---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-19T09:55:55.629844+08:00'
current: M24
next: register_full_model_direct_copy_validation_and_supervise_Object
execution_status: direct_Torch_block_copy_finite_reproduction_complete_full_model_pending_Object_running
---
# 当前研究进度

本机先交付直接Torch复制基线：三个完整定义源码一致，六真实fixture/18次数值和有效状态逐位通过，六点中位耗时接近原版；旧B0精度66通过/旧A/A性能待定保留；BAAI Goal/Spatial各1500局完成，Object14/数据准入/ON已完成，匹配T200与全500双卡运行。宏观completed不表示质量门通过。

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | 描述性完成，T483/C484/P473；配对FAILED/曝光/四false保留 | [完整结果](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial固定链 | `in_progress` | Goal/Spatial各1500局完成；Object14补采/数据准入/ON已完成，匹配T200与全500已接续，每卡一worker | [晨间结果/修复](../../experiments/libero-four-suite-morning-20261019/RESULT.md) |
| `M18` | 迁移原件/环境、释放旧机 | `completed` | 四套资产落BAAI；AutoDL不再SSH、关机UNKNOWN | [迁移](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限证据，不外推 | [576检查/60门/420轮](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评测迁新机 | `completed` | 原Object16有效，原新14完整但监督无效，今晨获准固定14补采；本机不自动接质量 | [资源接续](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE；旧失败/PARTIAL保留 | [旧报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | Torch对齐与有限性能 | `completed` | 108attention/54block/六full30/三policy严格数值过；旧计时有尾部与资源异常 | [结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；不再迁移开发，用户kill模拟器后已退出 | [安装结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 完整block直接Torch复现基线 | `in_progress` | 三个定义逐字复制，六fixture/18次数值与有效状态通过；六点中位耗时比0.9677–1.0033，单轮波动保留；旧66/旧A/A不改 | [复制结果；完整模型待验](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |

## 结果与边界

Goal T463/C464/P467，各500；Spatial T472/C466/P464，各500。800罚分均值Goal158.806/157.166/156.466、Spatial145.998/152.964/155.304。两套均有FAILED_FIRST_TWO_S_PAIRING，训练曝光/独立性与正式E2E未获解决；不把1500完整终态写成质量PASS。Long原结果和长尾见原汇总，不重算旧门。

用户明确先直接复现，已逐字复制完整block/attention/cache助手并提供可撤销模型入口。六真实action block0（B2/B14×短中长KV）18次逐位/有效KV/输入/当前RNG及历史gold通过；294完整block调用，worker约15秒自然退出。六点中位耗时比0.9677–1.0033，单轮比0.8587–1.0973，计时波动不写成加速。完整30层/机器人/原CUDA stride与alias仍未新增证明。旧B0两cache的66次精度、旧A/A2.512635%超2.5%的失败和NVML异常记录不改；旧多引擎watch已停止，fixed3 AA未运行，未为PASS重试。总412/5000，原10:48:46截止不变。

Object旧16有效，新14完整机器人终态但监督文件冲突无效；旧失败保留，用户今晨明确批准仅补14。v3首尾/物理卡/训练角色不变，T200先作为独立baseline并行，未用来重训ON。自动链/5分钟巡检见新health-v2，不声称待运行阶段已启动。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/paused72/default不改。原tile与保护源不动，不自动恢复120/独立TEST/新TF45chunk/撤回融合/架构搜索。详细预算/原截止见PLAN和各Experiment。

此前四文完整保留：[本次history](history/M24-before-GPU-followup-20261019/PROGRESS.md)。
