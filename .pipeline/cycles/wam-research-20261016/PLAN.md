---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T02:06:00+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与系统协同

当前M24按网页方案建立真实完整block顺序基线，先CPU；不是再做一轮attention小测或开启persistent scheduler。科研base固定ca5f37e，GPU新作业未授权。原M17质量队列继续。

| ID | 工作 | 当前状态与验收 |
|---|---|---|
| `M16` | Long500组三臂首轮 | COMPLETE_DESCRIPTIVE_NOT_QUALIFIED；T483/C484/P473，各500；[完整结果](../../experiments/long-morning-20261018/FINAL-SUMMARY.md)；配对FAILED/曝光/四false保留 |
| `M17` | Object/Goal/Spatial固定链 | RUNNING_REMOTE_AUTO_QUEUE；10/19 02:03 Goal336/500共同组、1010/1500局；Spatial340/500、1023/1500；Object余14自动等槽；[队列核验](../../experiments/libero-four-suite-20261017/QUEUE-HANDOFF-20261019.md) |
| `M18` | 迁移原件/环境、释放旧机 | COMPLETE_VERIFIED；四套资产落BAAI，AutoDL不再SSH、关机UNKNOWN |
| `M19` | 原算子数据CPU复核 | COMPLETE_LIMITED_SCOPE；[576检查/60性能门/420轮](../../experiments/operator-parity-evidence-20261018/REPORT.md)，不外推 |
| `M20` | 本机评测迁新机 | COMPLETE_RESOURCE_RELOCATION_QUEUED；Object16/30自然CLEAN，余14/ON/final500已排队，匹配T200成本另记 |
| `M21` | 原始/B14短中长矩阵 | COMPLETE_WITH_STANDALONE_FULL30_FAILURE；[旧统一报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)，旧失败/预算保留 |
| `M22` | Torch对齐与有限性能 | COMPLETE_LIMITED_ALIGNMENT_WITH_LATENCY_TAILS；108attention/54block/六full30/三冻结policy严格数值通过，54点性能有尾部；[结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | COMPLETE_INSTALLED_LOGIN_PENDING；0.162.1于~/cxz，未迁凭据/最新开发树；用户取消进一步迁移；[结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 完整block显式顺序B0 | COMPLETE_CPU_BASELINE_GPU_PENDING；六真实fixture和合法写CPU通过，最终源CPU复核已完成、GPU未运行；[本轮计划](../../experiments/wam-block-baseline-20261019/PLAN.md) |

## 可见计划

- [x] M16/M18/M19/M20原交付；M21负结果、M22有限数值和计时、M23安装及main镜像保留。
- [x] M24固定源码与已有fixture核查，独立B0、CPU契约/真实完整block与合法写验证。
- [ ] M24收口最终CPU版本、契约/coverage/待运行计时/资源来源包，HW和main同步，9点前交结果。
- [ ] M17原队列继续、5分钟低开销巡检；不恢复本机模拟器/质量任务，不自动重试。
- [ ] 获新明确GPU授权后才做B0最小数值/状态、A/A与配对计时；不借M22余量、不先开发scheduler。
- [ ] T25/T17对T50、后两轮Long的seed/预算另登记，未自动运行。

## 预算与长期边界

M24 CPU≤2、worker累计≤8h、新增≤1GiB、GPU0；08:30停新、08:50清理、2026-10-19 09:00汇报。GPU120min只是建议，未授权；不加载完整模型/创建CUDA context，不训练/新episode/迁移/生产patch。

M16原10/17 13:55:56起，10/19 13:40:56停新/13:55:56清理，6001000尝试/10GiB，实际1593717。M17原10/17 17:11:31起，10/21 16:56停新/17:11清理，18723000尝试/15GiB；Object phase原16:45停新/17:00清理、801000含旧296；迁移匹配T200另记800000，不增加独立样本、不重置起点。

M21旧1649.274/1800秒、约397MiB，r1/r2/r4/r7失败、r3等待超额与r5 PARTIAL保留。M22累计1094.781/1200秒、1002 full target forwards、406.87MiB，自有退出；CUDA active UNKNOWN，不挪余105秒。完整账本见对应Experiment。

所有评测用512-token BF16优化reset，cold另列；horizon800、actual796与罚分800分开。原tile代码/HW、state/cycle/rules、旧FAILED/retention/22缺收据、4f04 paused72/96与MLP0、本机live77、全局A_off及四false不改。新机写入限~/cxz，不杀他人/改系统，无smoke/模型hash。论文正文逐段交飞书；main仅进度镜像。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。
