---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-19T02:20:56+08:00'
current: M24
next: new_authorized_GPU_block_window_then_AA_and_profile_keep_remote_queue_running
execution_status: CPU_B0_and_22_contract_tests_complete_GPU_pending_remote_chains_running
---
# 当前研究进度

M24已建立B0_HYBRID_SEQUENTIAL；六个保存的真实action block0输入、六个B2合法写控制CPU逐位/有效状态/RNG通过；两fixture各28观察阶段/静态投影控制通过。最终源同集合复核已完成、22项CPU契约通过；GPU精度/性能未运行。见[本轮检查点](../../experiments/wam-block-baseline-20261019/CHECKPOINT.md)。

| ID | 状态 | 当前证据与next |
|---|---|---|
| `M16` | COMPLETE_DESCRIPTIVE_NOT_QUALIFIED | Long各500，T483/C484/P473；[汇总](../../experiments/long-morning-20261018/FINAL-SUMMARY.md)，原科学门未通过 |
| `M17` | RUNNING_REMOTE_AUTO_QUEUE | 10/19 02:03 Goal336/500共同组、1010/1500局；Spatial340/500、1023/1500；两卡各两worker，Object30秒自动等槽；[核验](../../experiments/libero-four-suite-20261017/QUEUE-HANDOFF-20261019.md) |
| `M18` | COMPLETE_VERIFIED | 原件/环境/四套权重落BAAI，旧AutoDL关机UNKNOWN、不再SSH |
| `M19` | COMPLETE_LIMITED_SCOPE | [576检查/60门/420轮复核](../../experiments/operator-parity-evidence-20261018/REPORT.md)，原证据不外推 |
| `M20` | COMPLETE_RESOURCE_RELOCATION_QUEUED | Object16/30自然CLEAN，余14/训练/全500已排队；本机不自动接质量 |
| `M21` | COMPLETE_WITH_STANDALONE_FULL30_FAILURE | [旧报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)，旧standalone完整六query非逐位/三例容差失败保留 |
| `M22` | COMPLETE_LIMITED_ALIGNMENT_WITH_LATENCY_TAILS | 去fast_math后108attention/54block/六full30/三冻结policy严格数值通过；54新计时点中block9/54、full30两例>5%慢；[结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | COMPLETE_INSTALLED_LOGIN_PENDING | 0.162.1已装、登录待用户；开发不再迁移，模拟器由用户kill后已退出；[结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | COMPLETE_CPU_BASELINE_GPU_PENDING | 固定ca5f37e；B0是新Python阶段组织、Torch原基元；22契约通过、GPU入口已准备但未运行；[CPU计划/来源](../../experiments/wam-block-baseline-20261019/PLAN.md) |

## 已知边界

M24没有新GPU、模型GPU加载、simulator/episode/训练。保存时已contiguous，原CUDA stride/alias/RNG未持久化；CPU原版与旧GPU gold不逐位，不拿CPU通过替GPU证明。旧54个Torch block的三轮max/min有45点>1.05、36点>1.10，最大7.387；仅证实波动，根因UNKNOWN，下一轮先A/A。

M22实际替换仍仅SDPA，投影/norm/FFN等Torch；完整模型/冻结policy通过不等于全设备自有化或机器人质量通过。M21旧失败不追溯修改；G1仅kernel证据、shared/guarded旧六query有限状态逐位，不外推。

Long SR96.6/96.8/94.6%，800罚分均值285.858/282.588/293.270；共同成功长度C/T P50/P90/max=1/1.073561/2.329167、P/T=1/1.054054/3.223577，>1.30为6/472、7/464。T→C丢11救12，T→P丢19救9；全部自然失败保留。C36/P66首两S严格配对FAILED，50init训练/DEV曝光，不是独立TEST。273组暖episode探索性T/C约1.4974、T/P约1.2330，非独占/cold E2E。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。TF_ON/PF2旧NO_GO、retention、paused72/MLP0、默认A_off均不改。无新120/TEST/正式45chunk/撤回融合/架构搜索/第二环境；后续GPU需新范围、窗口、预算。原预算截止见PLAN，不借余量/迁移重置。

完整先前四文见[本次history](history/M24-before-semantic-update-20261019/PROGRESS.md)，当前只保留语义导航。
