# 当前研究恢复入口

更新时间：2026-10-19T00:23:36+08:00 北京。先读[INDEX](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再聚焦[当前Cycle四文](cycles/wam-research-20261016/PROGRESS.md)。原保护源、旧失败和原tile保留，当前不全量展开旧索引。

## 最新交付：M23新机Codex安装

用户随后取消进一步开发迁移，本机遗留模拟器由用户自行kill；只读报PID，既定远端评测不变。

[新机比较与安装](experiments/remote-codex-setup-20261018/RESULT.md)：Codex0.162.1入口`~/cxz/bin/codex`，CODEX_HOME及缓存/临时均限~/cxz；版本/帮助通过，登录待用户，GPU0。新机45核等效/180GiB/双同型号A100，CPU余量明显优于本机16线程拥塞；未测工作负载倍数，不混两机正式时间，尚未迁最新算子/会话。

## M22真实Torch算子对齐

用户已授权我主要负责实际对齐，允许复用固定Torch/FlashAttention源码；独立v2不改旧tile。新版本仅移除--use_fast_math，179个vendor与去命名空间后的CUDA源逐字相同。本地CPU无GPU编译完成，仍为nvcc12.4/Torch wheel12.6，不声称同SASS。

[登记与可见计划](experiments/operator-torch-alignment-20261018/PLAN.md)；[九真实输入结果](experiments/operator-torch-alignment-20261018/NINE-INPUT-RESULT.json)：B2 video/action、B14 action各三KV，9/9输出逐位、输入布局/字节/RNG通过；旧同输入仅3/9逐位。新完整54点、六full30及三组完整canonical policy配对冻结重放**全部完成且有限逐位/状态通过**；原失败仍是旧版本的真实结果。

M22数值阶段用GPU生命周期615.057秒、714模型forwards、两核CPU、新增源/编译/证据约40.71MiB≤512MiB；r1/r2均真实退出，CUDA active UNKNOWN。[最终结果](experiments/operator-torch-alignment-20261018/RESULT.md)：108attention/54block、六完整30层、三短中长完整policy配对全部逐位/有效状态通过，CPU27保存输出复核一致。没有机器人episode。用户新空闲窗口已完成性能：self/cross中位比1.00144/1.00193，block .98465（9/54慢>5%），六full30中位.99393、最大1.06654；CPU15同源检查全等。M22累计1094.781/1200秒，全部自有退出，约406.87MiB≤512MiB；空载NVML24–25%及少样本限制保留，不签普遍稳定对齐。见[新性能](experiments/operator-torch-alignment-20261018/performance/RESULT.md)，四false不改。

GitHub现仅[main进度镜像](https://github.com/qingyiyi/WAM-egale/tree/main)：本地当前代码/HW直接发布，旧commit/失败保留，不再自动建发布分支。最近整体审查快照ca5f37e已由main包含；[收拢记录](maintenance/github-main-20261018/RESULT.md)。

## M21完成的证据与负结果

[统一报告](experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)：54性能点、414数值比较、六B14 query×四引擎完整30层、9保存输入I/O、24输出CPU复核。原始B2 video K272/1136/2144，action及B14 K288/1152/2160；H24/D128。B14仍七候选各Q16、CFG两组、各自private KV。

旧standalone单block容差过≠逐位；完整30层0/6逐位、3/6旧容差过，有效CFG max_abs .1875。shared/guarded仅替self、cross仍Torch，六query输出/输入/有效KV/RNG逐位过；仅有限输入，不代表接受/scheduler/完整任务。G1只有kernel证据。

旧性能self/cross Graph-core比例中位数.9812/.9817；standalone block14/54慢、范围.2465–3.0713，NVML空context非零未解，仍探索性。新性能实际20:14:26结束；后续共享仅验证。M21累计1649.274/1800秒、新增约397MiB≤512MiB、所有自有GPU已退出；失败/原PARTIAL全部保留。旧[有限基线复核](experiments/operator-parity-evidence-20261018/REPORT.md)576检查/60性能门/420轮不外推。

## Long首轮与远端队列

[Long最终500组](experiments/long-morning-20261018/FINAL-SUMMARY.md)：10任务×50init，T/C/P各500，成功483/484/473（96.6/96.8/94.6%），罚分均值285.858/282.588/293.270，实际命令142861/141230/146527。T→C丢11救12，T→P丢19救9。共同成功长度type7 P50/P90/max：C/T=1/1.073561/2.329167，P/T=1/1.054054/3.223577；>1.30为6/472、7/464。全部自然失败保留。

C36/P66首两S严格配对FAILED，50init有训练/DEV暴露；不是独立TEST/非劣结论。新机273组并行暖episode探索性T/C约1.4974、T/P约1.2330，不是独占/cold E2E。三轮是偏好，后两轮seed/预算未冻，不声称已完成。

M17截至10/19 00:18：Goal276/500共同组、857/1500局，Spatial276/500、862/1500。GPU0两Goal、GPU1两Spatial，99–100%；Object余14/ON/匹配T及全500已自动排队、30秒轮询任一自然空槽，5分钟CPU巡检正常。无新人工提交必要；[实测接续](experiments/libero-four-suite-20261017/QUEUE-HANDOFF-20261019.md)。本机模拟器已退出，开发不再迁移；原截止/四false不改。

M16原10/17 13:55:56起、10/19 13:40停新/13:55清理，6001000尝试/10GiB；实际1593717。M17原10/17 17:11:31起、10/21 16:56停新/17:11清理，18723000尝试/15GiB；Object迁移匹配Target200另记800000，不重置起点、不增独立样本。详见Cycle PLAN。

## 科学门与固定接口

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧TF_ON/PF2 NO_GO、retention、22缺收据、4f04 paused72/96与MLP0、本机live77、全局A_off不改。无新120/独立TEST/TF正式45chunk/撤回大融合或架构搜索。

TF64=53070参数，16token×17feature(state7/previous7/σ/nextσ/t÷1000)，单层pre-norm D64/H4/FF256，GELU/drop0/学习位置/双7残差头；无图像/prompt cross-attn。TF_ON3000/normalizer/tau=.007839680526450805、absΔσ×L2、K2B2/bonus/prefix_then_cost、BF16方向/FP32评分固定。PF2=SST完整policy请求，reset清零/cache不计，T50更新/51调用含终端；不是bonus或状态恢复。

所有评测用512-token BF16等价reset，冷启动另列；horizon800，actual796和罚分800分开。无smoke/新SHA；不杀他人、改系统/保护源，不删自然失败。T25/T17对T50、video20同网格的强对照需要新登记；90权重仍BLOCKED。

HW固定v15/legacy不升级或全量sync。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；正文逐段会话交飞书，不建本地论文初稿。旧长compact逐字保留M22/PROGRESS.compact-before-M22.md。
