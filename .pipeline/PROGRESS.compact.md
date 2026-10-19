# WAM研究短入口

更新时间：2026-10-19T09:39:02.089869+08:00。当前Cycle `wam-research-20261016`，当前M24；先按INDEX/CONTEXT-POLICY及约束记忆恢复，再读Cycle四文，不默认展开历史。

## 当前工作

- 本机：昨夜完整顺序B0 CPU六真实fixture+六合法写控制、36完整执行、两fixture各28阶段及22契约通过。今晨获新GPU授权，六真实action block0（B2/B14×KV288/1152/2160）×cache开关共66 GPU严格数值/有效状态/输入/当前RNG与旧GPU gold通过。B0是显式Python阶段+Torch原基元，不是全自有CUDA或persistent scheduler。
- 性能：今晨A/A最大2.512635%超2.5%即停；本机0MiB/无进程却NVML24–25%，根因UNKNOWN。两worker118block已自然退出。CPU监控每60秒待两次≤5%/0MiB/无进程后最多再启一轮，A/A门不改，原08:48:46起/10:48:46清理、单A100/CPU≤2/≤5000block/≤1GiB。计时仍PENDING，不签全形状对齐。[结果](experiments/wam-block-gpu-followup-20261019/RESULT.md)。
- BAAI：Goal/Spatial各500组三臂全齐，1500完整局/套。Target/TF/TF+PF2成功分别Goal463/464/467、Spatial472/466/464。800罚分均值分别158.806/157.166/156.466、145.998/152.964/155.304。已回收逐局报告并CPU独立核对唯一行、成败、步数、罚分/胜负；不是独立TEST或质量PASS。
- Object：旧16条有效；迁移续采新14虽有完整机器人终态，但编号重启冲突导致监督无效，原PACK_CURRENT拒绝入训保留。用户今晨明确批准只补这14局/最多56000次，在独立v3namespace隔离compact子目录。固定14补采/准入/ON已完成（FIT6164/CAL3080），原匹配Target200和全500双卡运行；09:21为T20018局、首FINAL5局，每卡一worker。更多并发槽的GPU PID/容器PID身份未统一，保留原准入与UNKNOWN。原无效14消耗5876已扣。v2过期模板启动失败0GPU/0episode保留；旧T_POOL截止不延长，新baseline沿原M17有效总截止。[两机晨间结果](experiments/libero-four-suite-morning-20261019/RESULT.md)。
- 远端每5分钟CPU只读巡检已切health-v2、真实出生身份匹配；新机无需Codex才能自动排队，后台是已登记Python controller/guard。旧watch正常交接，不杀他人。

## 稳定证据与限制

Long首轮各500完整：T483/C484/P473，SR96.6/96.8/94.6%；800罚分均值285.858/282.588/293.270。C/T共同成功P50/P90/max=1/1.073561/2.329167、P/T=1/1.054054/3.223577；>1.30为6/472、7/464。丢失/救回T→C为11/12、T→P为19/9。首两S严格配对C36/P66 FAILED、50init训练/DEV曝光；旧273组warm episode探索性比约1.4974/1.2330，不是独占/cold E2E。[Long结果](experiments/long-morning-20261018/FINAL-SUMMARY.md)。

Goal/Spatial也有FAILED_FIRST_TWO_S_PAIRING：Goal C165/P334、Spatial C155/P312。这些是组件配对失败数，不是额外机器人自然失败；保留全部结果，不用成功子集/旧target替代新配对。两套共同成功C/T长度max2.6667/2.7905，P/T4.8889/4.1563，长尾未解决。性能原质量并行计时只探索性，正式E2E UNKNOWN。

M22仅attention核心去fast_math版本：108attention、54block、六full30和三冻结policy有限逐位/有效状态过。54新性能点9点block>5%慢、full30两点>5%，不保证全形状性能；M21旧standalone full30失败和全部PARTIAL保留。B0完整30层/policy/GPU合法写控制尚不继承CPU或M22证书；原CUDA stride/alias/RNG未持久化，video完整block缺fixture。[算子基线](experiments/wam-block-baseline-20261019/RESULT.md)。

## 不变的门与权限

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧TF/PF2 NO_GO、retention、paused72/96与MLP0、22缺收据、默认A_off不改。TF64/TF_ON接口及权重身份、tau0.007839680526450805、K2B2/bonus/prefix_then_cost、BF16模型/FP32打分不变。B14=CFG2×7个各Q16状态，不是Q112互相注意；PF2为SST完整policy，T50更新/51action调用含末尾，不是bonus或物理恢复。

不自动120/独立TEST/新TF正式45chunk/架构搜索/第二环境/撤回融合。原tile代码/HW与state/cycle/rules保护；保留FAILED/自然失败/来源，不挑seed/改阈值放行。所有任务同等512-token BF16优化reset，cold另列；actual796和800罚分分列。新机所有写入只限~/cxz，AutoDL不再SSH、关机UNKNOWN，无大文件hash/smoke，不杀他人/改系统。

M16原10/17 13:55:56起至10/19 13:55:56清理，Long已完，不借余量；M17原10/17 17:11:31起，10/21 16:56:31停新/17:11:31清理，18723000尝试/15GiB；原Object/补采/匹配T成本都累计，不重置。GitHub仅main当前公开镜像，旧固定commit身份保留；最新发布以实际receipt为准。中文论文逐段会话交飞书，不建初稿。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。

详情只按需读[Cycle进度](cycles/wam-research-20261016/PROGRESS.md)、[计划](cycles/wam-research-20261016/PLAN.md)与当前Experiment。历史四文和所有原证据均保留；新结果不覆盖旧门。
