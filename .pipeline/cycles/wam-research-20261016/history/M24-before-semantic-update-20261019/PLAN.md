---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T00:23:36+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与系统协同

Long首轮已交；四套既定评测继续。最新用户要求本机优先算子研究、质量评测迁新机；已授权短中长原始/B14补测。M21旧性能/共享精度证据含standalone完整30层失败，保留。M22按新授权独立去fast_math修正已通过有限完整模型/policy逐位与状态核验，新二进制有限性能已测，慢点与资源限制保留。旧科学门与默认不变。

| ID | 工作 | 当前状态与验收 |
|---|---|---|
| `M16` | Long500组三臂首轮 | COMPLETE_DESCRIPTIVE_NOT_QUALIFIED；T483/C484/P473，各500；原配对FAILED/曝光/四false保留；[最终](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial固定链 | RUNNING_REMOTE_AUTO_QUEUE；10/19 00:18 Goal276组/857局、Spatial276/862；Object余14自动等槽；[核验](../../experiments/libero-four-suite-20261017/QUEUE-HANDOFF-20261019.md) |
| `M18` | 迁移原件/环境并释放旧AutoDL | COMPLETE_VERIFIED；四套资产及原件落新机，旧实例不再SSH，是否关机UNKNOWN |
| `M19` | 原算子既有数据CPU复核 | COMPLETE_LIMITED_SCOPE；576检查/60性能门/420轮、180源/二进制逐字相同；[报告](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评测迁新机，留本机供算子 | COMPLETE_RESOURCE_RELOCATION_QUEUED；Object16/30自然暂停CLEAN，余14/ON/全500已迁队列；匹配Target200迁移成本另列 |
| `M21` | 原始/B14三档KV性能与完整I/O | COMPLETE_WITH_STANDALONE_FULL30_FAILURE；54性能点、414数值比较、6×4完整query、9保存输入；[统一报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 独立Torch对齐、真实性能与GitHub | COMPLETE_LIMITED_ALIGNMENT_WITH_LATENCY_TAILS；严格数值通过，54性能点+六full30已测，attention约1.002，block尾部保留；[新性能](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |

| `M23` | 新机资源比较与Codex安装 | COMPLETE_INSTALLED_LOGIN_PENDING；0.162.1入口`~/cxz/bin/codex`，GPU0；用户取消进一步开发迁移，模拟器自行结束；[报告](../../experiments/remote-codex-setup-20261018/RESULT.md) |

| `M24` | RUNNING_CPU_AUDIT_BASELINE_PREPARE | 固定ca5f37e，建立完整block R0/R1/B0，GPU未授权；[可见计划](../../experiments/wam-block-baseline-20261019/PLAN.md) |

## 可见计划

- [x] Long500原件与完整统计回传；质量与探索性成本分开。
- [x] 本机Object在完整episode后自然释放；迁出未开始评测和所需数据。
- [x] 新算子矩阵三种输入/三KV/first-last/三层，保存全部时间和误差。
- [x] 遵用户不足20分钟窗口，独立性能短协议于20:14结束；其后仅共享精度，未混计时间。
- [x] 全54输出/有效KV、完整30层六query及九保存输入验证；失败保留，自有全部退出。
- [x] M23两机只读核验、限~/cxz安装Codex、真实版本/登录状态与清理交付。
- [ ] M17原队列继续，结果齐后讨论；不恢复本机400局或新增Long训练。
- [x] M22独立去fast_math修正，真实完整模型/三完整policy配对精度与状态核验；HW/GitHub和网页prompt已交。
- [x] 用户确认新空闲窗口，M22新二进制54点/六full30实测完成，全部慢点及资源异常保留。
- [ ] 如继续算子工作，先定位block和完整模型慢点/计时波动；不再盲改数值，原本轮剩余预算不自动扩展。
- [ ] T25/T17对T50完整DEV另登记；后两轮seed/预算未冻结，不自动运行。

## 预算与边界

M16原10/17 13:55:56起，10/19 13:40:56停新/13:55:56清理，6001000尝试/10GiB含fixture3GiB；实际1593717，不重置。

M17原10/17 17:11:31起，10/21 16:56停新/17:11清理，原18723000尝试/15GiB；Object phase原16:45停新/17:00清理、801000含旧296。迁移同卡Target200新增成本800000尝试单列，非新增独立样本，见资源报告；原起点/截止不重置。

M21原1800秒登记生命周期实际1649.274秒（CUDA active UNKNOWN），新增约397MiB≤512MiB。旧r1/r2/r4入口/资源失败、r3等待超额收尾、r5用户收尾31点PARTIAL和r7准入预留错误全部保留。性能单进程3轮仅探索性；不签稳定性能或质量证书。

全评测使用512-token BF16优化reset，冷启动另列；horizon800，失败800罚分与实际796分开。原tile代码/HW、保护state/cycle/rules、旧FAILED/retention/22缺证/4f04 paused72/96与MLP0、全局A_off及四false均不改。新机仅~/cxz，旧AutoDL不再SSH，无smoke/SHA、不杀他人。

10/22 go/no-go，10/23冻结，10/28主表，11/01 18:00中文证据包。完整旧检查点保存在history，本次仅压缩派生导航。
