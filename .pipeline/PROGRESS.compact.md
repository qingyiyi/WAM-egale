# 当前研究恢复入口

更新时间：2026-10-19T02:06:00+08:00 北京。先读[INDEX](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再聚焦[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)。不默认展开旧索引/长JSON；保护源与原tile不动。

## 当前：M24完整block基线（CPU完成、GPU待验）

用户上传网页方案并要求同步HW、9点前交进展。科研base固定ca5f37e0482c2e38e3738e3d8e71e3b185990a23，本轮MODE=CPU_AUDIT_AND_BASELINE_PREPARE；新GPU/完整模型加载/模拟器/训练/episode/迁移/生产patch均未授权。CPU≤2、累计≤8h、新增≤1GiB；08:30停新/08:50清理。GPU120min只是下一阶段建议，不借旧M22余额。

[最终结果](experiments/wam-block-baseline-20261019/RESULT.md)：已实现独立B0_HYBRID_SEQUENTIAL，显式self/cross/FFN/norm/调制/残差/cache，原参数与Torch基元；不是全自有CUDA。六个真实action block0输入（B2/B14×短中长）、六同输入合法写控制、36次CPU完整执行逐位/有效状态/RNG通过；两fixture各28观察边界/静态投影控制通过，22项CPU契约/负例通过。仅六唯一输入，不是机器人样本。

已有GPU矩阵/专用worker代码，受新授权、源hash、六输入、local UUID/独占与预算约束，尚未执行。54性能单元均无数值；GPU精度/性能、完整模型/policy、原生产CUDA stride/alias/RNG及原始sigma/timestep仍有缺口，CPU通过不借作GPU证书。当前所有登记CPU退出，4个失败保留、首轮出生缺证不补造。[契约](experiments/wam-block-baseline-20261019/BLOCK-CONTRACT.md) · [来源](experiments/wam-block-baseline-20261019/SOURCE-EVIDENCE-PACK.md) · [预算](experiments/wam-block-baseline-20261019/BUDGET-AND-NEXT-STAGE.md)。

下一步唯一优先：新获批窗口下六fixture GPU数值/状态→A/A→匹配完整block计时；之后按profile选一个D0，暂不开发persistent scheduler。旧R0三轮轮内中位数max/min有45/54>1.05、最大7.387，根因UNKNOWN；不能拿旧少轮中位保证稳定对齐。

## M22既有有限对齐；M21旧失败保留

[新数值](experiments/operator-torch-alignment-20261018/RESULT.md)只去--use_fast_math，179 vendor逐字保持：108attention/54block、六30层冻结query、三短中长完整冻结policy数值/有效状态逐位通过；CPU27保存输出、15同源核对过。实际替换仅60个SDPA入口，投影/norm/FFN仍Torch，不能称全自有block或机器人证书。

[新性能](experiments/operator-torch-alignment-20261018/performance/RESULT.md)：self/cross Graph-core中位比1.00144/1.00193；block .98465但9/54慢>5%，六full30 .99393、2/6慢>5%，资源/NVML空载24–25%异常、少样本保留。累计1094.781/1200秒、1002模型forwards、406.87MiB，自有退出，不挪余105秒。没有新TF正式45chunk或任务E2E。

[M21旧报告](experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)：54点/414比较/六query×四引擎/九保存输入。旧standalone完整六query非逐位、三例旧容差失败（有效CFG max_abs .1875）；shared/guarded有限六query过，G1仅kernel。1649.274/1800秒、约397MiB，失败/PARTIAL全保留，不用M22新版本回写。原[576检查/60性能门/420轮](experiments/operator-parity-evidence-20261018/REPORT.md)也只限原范围。

## Long首轮与远端原链

[Long各500](experiments/long-morning-20261018/FINAL-SUMMARY.md)：T/C/P成功483/484/473，96.6/96.8/94.6%；800罚分均值285.858/282.588/293.270。共同成功长度type7 C/T P50/P90/max=1/1.073561/2.329167，P/T=1/1.054054/3.223577；>1.30为6/472、7/464。T→C丢11救12，T→P丢19救9。自然失败全保留。

C36/P66首两S严格配对FAILED、50init有训练/DEV曝光，不是独立TEST。新机273组并行暖episode探索性T/C约1.4974、T/P约1.2330，不是独占/cold E2E。用户偏好三轮，后两轮seed/预算未冻结，不自动启动。

M17最近快照：10/19 02:03 Goal336/500共同组、1010/1500局；Spatial340/500、1023/1500；Object=WAITING_NATURALLY_FREED_REMOTE_SLOT_CURRENT14。Goal/Spatial两卡各两worker；Object余14→ON→匹配T/全500自动等自然槽、30秒轮询，5分钟低开销CPU巡检持续。[队列核验](experiments/libero-four-suite-20261017/QUEUE-HANDOFF-20261019.md)，实时原件看`投机采样/results/runtime/p0-remote-server-health-cpu1-v1-20261018-r1/LATEST.json`，此摘要不是实时流。未向运行中评测注入本轮算子代码。

M16原10/17 13:55:56起，10/19 13:40停新/13:55清理、6001000尝试/10GiB，实际1593717。M17原10/17 17:11:31起，10/21 16:56停新/17:11清理、18723000尝试/15GiB；Object phase801000含旧296，迁移匹配T200另记800000、不增独立样本、不重置预算。细账见Cycle PLAN与Experiment。

## 长期边界与导航

四false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON/PF2 NO_GO、retention、22缺收据、4f04 paused72/96与MLP0、本机live77、全局A_off不改。无自动新训练/120/独立TEST/TF正式45chunk/撤回大融合/架构搜索/第二环境。

TF64=53070参数、16token×17feature、单层pre-norm D64/H4/FF256、GELU/drop0/学习位置/双7残差头，无图像/prompt cross-attn；target仍有视觉条件。TF_ON/normalizer/tau=.007839680526450805、absΔσ×L2、K2B2/bonus/prefix_then_cost、BF16方向/FP32分数固定。PF2=SST完整policy，reset清零/cache不计，T50更新/51 action调用含末尾；不是bonus或物理状态恢复。

所有既定评测用512-token BF16等价reset，cold另列；horizon800，actual796与罚分800分开。T25/T17对T50强对照另登记，video20同网格。90同代权重仍BLOCKED。

本机模拟器已由用户kill且退出，不恢复；本机开发不再迁移。BAAI写入永远限~/cxz、不拷SSH私钥/杀他人/改系统。M23 Codex0.162.1已装、登录待用户，未迁最新算子/会话；AutoDL不再SSH、关机UNKNOWN。无新smoke或模型大文件hash；本轮仅按附件做小型源码来源指纹。

GitHub只保留[main进度镜像](https://github.com/qingyiyi/WAM-egale/tree/main)，固定commit供单次审查，不以main覆盖科研base。HW固定v15，不升级/全量sync；原保护state/cycle/rules与旧长记录保留。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；论文正文会话交飞书，不建本地初稿。压缩前入口原文见本轮Experiment的PROGRESS.compact-before-M24.md。
