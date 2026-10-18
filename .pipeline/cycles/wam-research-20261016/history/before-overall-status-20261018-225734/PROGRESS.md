---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-18T22:39:41+08:00'
current: M23
next: user_terminates_local_simulator_keep_existing_work_allocation_and_remote_queues
execution_status: remote_codex_installed_auth_pending_no_GPU_work
---
# 当前研究进度

当前主任务M22已完成独立算子的有限严格数值对齐：去除fast_math后108attention/54block、六完整30层和三短中长完整policy配对均逐位/有效状态通过。[新结果](../../experiments/operator-torch-alignment-20261018/RESULT.md)。新二进制54点和六完整30层性能已在用户确认窗口补测，数值保持有限严格一致；block/模型存在慢点及资源计数异常，见[新性能](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md)，自有GPU已退出。旧M21完整30层失败属于旧二进制，原样保留在[旧统一报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)，不把新通过追溯改写旧失败。

| ID | 状态 | 当前证据与next |
|---|---|---|
| `M16` | COMPLETE_DESCRIPTIVE_NOT_QUALIFIED | Long T483/500、C484/500、P473/500；[完整汇总](../../experiments/long-morning-20261018/FINAL-SUMMARY.md)，旧质量NO_GO/四false保留 |
| `M17` | RUNNING_REMOTE_MONITORED | 22:14 Goal208共同组/690局、Spatial202/683；Object余14仍排队，controller/worker存活；5分钟CPU巡检；[快照](../../experiments/libero-four-suite-20261017/PROGRESS-20261018-2214.md) |
| `M18` | COMPLETE_VERIFIED | 原件/匹配环境/四套资产落BAAI；旧机清理核验、是否关机UNKNOWN，不再SSH |
| `M19` | COMPLETE_LIMITED_SCOPE | [原数据复核](../../experiments/operator-parity-evidence-20261018/REPORT.md)：576检查、60性能门、420轮，原有限证书不外推 |
| `M20` | COMPLETE_RESOURCE_RELOCATION_QUEUED | Object16/30完整后CLEAN，controller/server/旧handoff退出；余14/ON/final500迁新机，本机不自动接续质量任务 |
| `M21` | COMPLETE_WITH_STANDALONE_FULL30_FAILURE | [当前结果](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)：54性能点、414数值比较、24完整model执行、27保存输入I/O比较；全部自有进程已退出 |
| `M22` | COMPLETE_LIMITED_ALIGNMENT_WITH_LATENCY_TAILS | 数值有限逐位通过；新54性能点+六full30完成，attention中位比约1.002、block9/54及full30两例慢>5%；全部自有退出，[结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |

| `M23` | COMPLETE_INSTALLED_LOGIN_PENDING | 用户取消进一步开发迁移；新机资源已核验；Codex0.162.1与入口已实测，未迁凭据/新算子；[安装结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |

## M21旧二进制实测（保留负结果）

54点覆盖B2 video/action及B14 action、短中长、first/last、层0/8/29；video K272/1136/2144，action K288/1152/2160，B14是七个独立Q16/CFG2。单block容差不等于逐位；standalone完整六query均非逐位、三例旧容差失败，有效CFG max_abs .1875、rel_L2 .03512。shared/guarded仅替self、cross仍Torch，有限六query全部有效状态逐位；不外推闭环。

旧性能为单进程三轮探索性，standalone block14/54慢，NVML异常未解；20:14性能结束后仅共享精度，不混计。累计1649.274/1800秒、新增约397MiB、全部自有退出；r1/r2/r4/r7失败、r3等待超额与r5 PARTIAL均保留。完整数字/输入/误差/轮次见[旧统一报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)，压缩前全文见本次history。

## Long500首轮边界

T/C/P成功483/484/473，SR96.6%/96.8%/94.6%；800罚分均值285.858/282.588/293.270，实际命令142861/141230/146527。T→C丢11救12，T→P丢19救9。双方成功长度type7 P50/P90/max：C/T=1/1.073561/2.329167，P/T=1/1.054054/3.223577；>1.30为6/472、7/464。

C36、P66原首两S严格配对FAILED，50init有训练/DEV暴露，不能签独立TEST。新机273组暖episode探索性全尝试T/C约1.4974、T/P约1.2330，非独占/cold E2E。所有自然任务失败保留，原PARTIAL补齐不重复计样本。

## 当前权限与下一步

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission四项false。TF_ON/PF2旧NO_GO、retention、4f04 paused72/96、MLP0、本机live77、全局A_off不改。无新45chunk/TEST、融合、训练或第二环境。

用户要本机研究算子，评测尽量在BAAI；原tile代码/HW不动。原M16/M17预算截止见PLAN，不因迁移或补缺重置。M22已完成单因素数值修正；新二进制有限性能已测，后续先定位慢点；M17原队列继续。M23只安装Codex，登录及最新开发树迁移未完成。新GPU作业需有明确范围/窗口/预算，不借未用余额自动扩任务。
