---
kind: plan
cycle: wam-research-20261016
status: active
updated: 2026-10-17T17:22:00+08:00
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与系统协同

当前M17：用户新增批准Spatial/Object/Goal各自同配方TF64训练与Target/TF/PF2全套描述评估；M16 Long继续，不重置其48h预算。具体执行以[三套PLAN](../../experiments/libero-four-suite-20261017/PLAN.md)及[实际作业](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)为准。

## 可见执行状态

- [x] 三个社区Target固定发布revision、norm_stats与非官方身份核查；官方90 Target仍BLOCKED。
- [x] 独立完整任务server/client、collector、固定TF训练、预算guard与campaign源码冻结；只做静态审查，没有smoke/独立测试/SHA。
- [x] Object/Goal远端权重完整；Object正回传。完整512-token BF16 bank正在CPU生产。
- [x] 新96h master与三个控制器登记/启动，等待资产与资源；目前新增机器人局0，不能把controller等同model/episode已启动。
- [ ] Object本机完整Target200→固定训练链→最终500组三臂；Goal/Spatial等待Long自有进程清理后在AutoDL接续。
- [ ] 全成败、罚分、配对差、长度及探索性成本汇总；质量和无争用放行后才正式完整任务E2E。

## 冻结与止损

每套1560新增完整episode：T_POOL200复用其中Target评估；两次C30为训练采集；最终T/C/P各500局。三套4680局、18,723,000 target尝试、研究证据15GiB；固定一次PARENT2000→CHILD2500/CONT3000采集→ON3000，从同CHILD起点训练ON，不做seed/结构/阈值搜索。FIT/CAL按整局隔离，全部50init含训练暴露，不签独立TEST。

同case全部臂同物理GPU，最终每套分两个预登记250case shard；可同卡两个独立model，CPU/显存准入、环境/RNG/KV/端口隔离，卡上他人进程不动。所有对照默认完整优化reset。budget、等待、失败、超时、清理均记账；M17最迟10/21 17:11清理，旧Long最迟10/19 13:55仍独立。实施失败停止当前chain，不自动重试，不删自然失败。

四权限false、旧NO_GO/retention/FAILED/22缺证、全局A_off、原tile与保护state/cycle/rules均保留。无120/独立TEST/新TF正式45chunk；replay不能替代完整episodeE2E。原tile不动，不恢复撤回融合，当前优先draft。

10/22质量/系统go-no-go；10/23冻结候选/API/claim；10/28主表；11/01 18:00中文证据包。中文稿会话逐段给用户整理飞书。

M01–M16完整计划及旧验收边界见[原计划归档](history/PLAN-through-M16-20261017.md)，原始结果路径不变。
