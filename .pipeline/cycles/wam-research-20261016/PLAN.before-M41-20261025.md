---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-25T00:52:21.577061+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前计划：逐阶段优化与证据收口

- [x] M39-SM120-FFN-NEXT：执行用户新附件T0–T3；[有限接续计划](../../experiments/sm120-ffn-next-20261024/PLAN.md)，当前SM120独立A、七段BF16递推、成本后选择至多一个候选/一次修订；逐项完整表，原资源/失败/科学门保留。

- [ ] H02-LOCAL-STORAGE：按用户新资源分工进行本机轻量化；先统一/rjs/cxz/tmp并清理闲置/tmp，再处理逐字节已确认的13.21GiB公共组件重复，逐文件校验落盘确认后立即释放大文件，不等整批。CPU≤2、GPU0、不改机器人队列/投机源码/保护文件；记录真实回收量和恢复映射。见[维护计划](../../maintenance/storage-placement-20261024/PLAN.md)。

- [x] M39-FFN-HANDOFF：已授权实际试验输入/跨任务写回、小ACT及双CTA方案；[本轮计划](../../experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/PLAN.md)，先能力/有限精度后两进程整层，逐候选全表；现有5090持续授权、原账本与双A100不改。

- [x] M39-SELF-QKV-TMA：用户已追加现有5090持续使用授权。将当前W0 TMA／W1–3 epilogue／W4–7 MMA迁移至Self Q/K/V；保留288任务、64×128 tile、十段K与舍入，独立A及候选先逐位后两进程三KV计时，公开入口两流核验；当前job限20分钟、4CPU、1GiB。原十小时账本不改，A100 RoboTwin继续原160小时预算。

- [x] M39-CURRENT-MMA-MAINLOOP：用户已同意拆解当前四MMA投影的输入/操作数/主循环；独立观察副本、一次构建，最多250秒且原累计余量253.587秒不增。数值和粗细插桩扰动后解释，保留完整历史阶段表。

- [x] M39-RESULT-EMPTY：用户要求先检查当前result_empty等待及TMA写回边界；仅独立观察副本、一次构建、三KV完整block，最多420秒且原累计不增。[诊断计划](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/result-empty-diagnostic/PLAN.md)。

- [x] M39-PROJECTION-WARP-SPECIALIZED：用户新批准四MMA＋一TMA＋三epilogue，仅三个投影；[有界计划](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/PLAN.md)，25分钟且旧累计不增。

- [x] M39-TMA-INPUT-HANDOFF：用户同意优先处理TMA发起与输入交接；[CPU源码核查与最小改法](../../experiments/rtx5090-block-port-20261023/projection-tma-components/INPUT-HANDOFF-SOURCE-REVIEW.md)已完成。八MMA warp/数值顺序保持，暂缓输出分工；GPU实现与验证未启动，旧剩余30.2分钟不增。

- [x] M39-PROJECTION-TMA-COMPONENTS：用户批准先分量诊断/扰动核验，35分钟、累计不增；[计划](../../experiments/rtx5090-block-port-20261023/projection-tma-components/PLAN.md)。

- [x] M39-PROJECTION-TMA-CONTINUOUS：用户批准先试跨十数值段的连续TMA输入ring，原MMA/后处理；[50分钟有界计划](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/PLAN.md)。

- [x] M39-PROJECTION-SPLITK：用户批准三个小投影有序K分段，60分钟且原累计不增；见[计划](../../experiments/rtx5090-block-port-20261023/projection-splitk/PLAN.md)。

- [x] M39-FFN-SPLITK：用户批准先试FFN2跨CTA的K分段计算及有序BF16提交，原主循环/tile不变；[有界计划](../../experiments/rtx5090-block-port-20261023/ffn-splitk/PLAN.md)，90分钟且累计不增。

- [x] M39只读诊断续项：核对GRAIN/TMA主循环、96tile并行度、分段BF16提交和官方SM120 BF16 TMA参考，给出FFN/小投影复用方案；[源码分析](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/TMA-GEMM-DIAGNOSIS.md)。无新GPU任务，实施候选尚未启动。

- [x] M39-FFN-TMA-PIPE：最新用户批准分步实现；[有界计划](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/PLAN.md)，当前采用GRAIN，原累计预算不增加。

- [x] M39-SM120-TMA：用户批准基于SM120能力的有界优化，先FFN2 TMA；[计划](../../experiments/rtx5090-block-port-20261023/blackwell-tma/PLAN.md)。

- [x] M39-RTX5090-EAGER-STAGES：补当前精度版三方eager、融合阶段加速比与完整层；[计划](../../experiments/rtx5090-block-port-20261023/eager-stage-comparison/PLAN.md)。

- [x] M40-DUAL-5090-HANDOFF：用户新分工已生效，远端双A100接续固定RoboTwin队列；算子/性能统一5090，恢复三方阶段表，原预算不增。[交接结果](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-continuous-20261023/RESULT.md)。

- [x] M39-RTX5090-PRECISION：新用户授权先修5090精度；两小时上限、旧累计不重置；[计划](../../experiments/rtx5090-block-port-20261023/precision/PLAN.md)。

- [x] M39-RTX5090-PORT：用户新授权现有两槽算子移植SM120；三KV原Torch对照、完整阶段表、源码/结果回本机；2小时上限含原累计，详见[计划](../../experiments/rtx5090-block-port-20261023/PLAN.md)。

- [x] M39-RTX5090-SETUP：用户批准隔离配置Torch2.9.0/CUDA13.0并尝试实例重启；60分钟/4CPU/20GiB/GPU kernel0，给出计数器命令并回传安装记录；不改驱动或恢复旧研究窗口。

- [x] M39-RTX5090-ENV：按新机授权只读核查GPU/资源/软件/计数权限；保存条件选卡与回同步要求。未运行kernel/编译/软件安装；SM120环境、占用原因和源码适配未完成，不恢复旧GPU窗口。

- [x] M39-FFN1-TWO-STAGE：按用户新要求仅将当前L13研究worktree的FFN1尾块输入缓冲三级改两级；CPU≤2、编译≤10分钟、新增≤512MiB、GPU0秒。核对源码差异与完整kernel编译/结构尺寸；旧T0–T4闭合与GPU账本保留，性能和精度待后续验证。

- [x] M39-MOD：按需调制有限验证与采用，历史结果保留。
- [x] M39-RR：用户批准self RMS+RoPE；实现、编译、9签名/21结构、三KV整层计时完成，两机采用。
- [x] M39-TC：核对投影及attention QK/PV的源码与活动二进制Tensor Core指令；解释RoPE与MegaMoE固定角色。
- [x] M39-RECORD：优化索引、版本身份、失败及资源归档；owned清理。
- [x] M39-KV-AUDIT：按用户新顺序核查Cross K/V分组复用；源码与三档压缩fixture支持每层两CFG组，CPU完成；online RMS文献及当前实现核对完成。
- [x] M39-KV-IMPL：已获准在optimized实现显式context分组，保持独立RMS；CPU审查与编译。
- [x] M39-KV-GATE：真实9签名及结构/条件换序核验，有限逐位通过才进入计时。
- [x] M39-KV-PERF：三KV两进程完整层、强baseline与复用边界；更新阶段表/证据/owned清理。
- [x] M39-VA2：核查LingBot-VA2原文加速基线及推理框架。
- [ ] M39：按最新决定恢复Torch/guarded/自有三方同条件eager；SM120两候选已闭合，采用M64，TMA退化；下一步讨论剩余FFN2及小投影。旧仅Torch展示规则已被替代。
- [ ] M17/M40：M17保留原证据缺口；M40现为双A100持续接续，原160小时/10天不变。10点临时窗口为历史。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)。累计18427.875/36000秒；当前作业闭合，原截止与四科学权限不变。

- [x] M39-KV-CACHE：用户新批准自有算子读取预计算K（含RMS）/V，双方同等缓存；先精度后性能，冷构建另列。
- [x] M39-KV-STAGES：完整层、逐阶段耗时、同步等待分列；profile与正常计时独立。
- [x] M39-TILE-TABLE：按用户反馈补齐阶段tile数量、每tile形状/工作量、K分段与attention循环口径；仅CPU源码核查及文档同步，不启动GPU。

- [x] M39-GELU-AUDIT：核对GELU 1568tile来源及FFN1/GELU/FFN2/残差依赖；仅源码/旧证据分析，融合与warp重叠未实现、无新计时。

- [x] M39-FFN-FUSE：B14两处写回融合已完成；9签名/36结构有限逐位及状态通过，原Torch三KV两进程整层对照完成。融合本身退化0.83%–1.14%，保留负结果和旧入口；本轮作业已清理。

## 稳定计划ID

M39子项沿用上方名称，以下只补齐语义导航，不重编号历史任务或新增授权。

| ID | 阶段 | 期望结果 | 验证方式 |
| --- | --- | --- | --- |
| `M39-FFN-SPLITK` | FFN2有序split-K | 有限逐位通过，整层改善4.47%–5.40%，已采用 | ffn-splitk/REVIEW_RETURN.md |
| `M39` | 单层算子对齐与优化 | 保留有限正确性、性能及负结果 | 当前FFN融合结果；后续先定位融合开销 |
| `M39-FFN1-TWO-STAGE` | FFN1两槽候选 | CPU编译/容量已核验，GPU未测 | 两次构建原件与sizeof结果 |
| `M39-RTX5090-ENV` | 新5090环境只读核查 | 条件选卡、软件/权限缺口、回同步要求 | maintenance/rtx5090-environment-20261023 |
| `M39-RTX5090-PRECISION` | 5090当前原Torch精度对齐 | 三KV26中间量＋输出、两进程状态通过；新版性能未测 | experiments/rtx5090-block-port-20261023/precision |
| `M39-RTX5090-PORT` | 5090完整层移植与三KV对照 | 有限精度、原Torch比值与阶段表 | experiments/rtx5090-block-port-20261023 |
| `M39-RTX5090-SETUP` | 项目内5090工具链安装 | Torch2.9/cu130、SM120编译工具与本机归档 | 无GPU计算；60分钟累计安装截止不重置 |
| `M40-DUAL-5090-HANDOFF` | 资源与对照口径交接 | 双A100已接续，5090三方阶段表 | 启动身份、真实动作、共享队列/预算与后台巡检 |
| `M17` | LIBERO证据收口 | 已有四套结果及未通过配对/独立性边界 | 原报告；不恢复过期队列 |
| `M40` | RoboTwin迁移与证据 | 固定1800局扩大DEV | 用户新分工为双A100接续，算子统一5090；原预算保留 |

- [x] M39-FFN-PIPE：四组FFN依赖流水完成；R2有限逐位/状态通过，相对此前阶段版整层小幅改善0.33%–0.36%。R1负结果、R2目录错误FAILED及同截止retry保留；owned清理。

- [x] M39-FFN-DIAG：三KV两进程FFN原Torch对照、整层/独立时间线、本机NCU及生产SASS定位完成；R2不改，自有清理，失败保留。

- [x] M39-FFN-TAIL：尾部四块拆为12个M32任务，有限门通过；原Torch三KV完整层1.1910/1.7241/2.7993×，本次延迟降约1.9%。两机研究采用，owned清理。
- [x] M39-FFN-EPILOGUE：A四元素写回、B生产函数内联均有限通过并提速；当前完整层原Torch比值1.2185/1.7605/2.8473×，独立FFN仍慢约9%；两机研究采用，owned清理。
- [x] M39-FFN-REMAINING：当前B时间线＋NCU完成，有限数值/状态通过；已定位末组生产延迟与额外local/指令，不改生产入口。

- [ ] M40-EVAL100：新160GPU小时/10天资源已授权；固定1800局，GPU1已补齐600有效初态与bank，固定质量队列已自动运行；最新两张A100持续用于RoboTwin，算子统一5090。

- [x] M39-FFN-OUTER-ABI：外层流水内联完成，有限门/整层对照通过并采用；后续尾K64也已完成。

- [x] M39-FFN-TAIL-K64：只改12个FFN1尾块K64，有限门及三KV两进程完整层通过，两机采用。
- [x] M39-CURRENT-STAGES：最终当前源码的计算/分块/warp/同步和原Torch阶段比值；GPU0两进程软件插桩测量已闭合，资源属性一致、扰动≤0.462%，原00:58截止保留。

- [x] M39-SELF-QKV-WARP：先核对四warp映射；仅Self Q/K/V固定64×128 tile试8warp，有限精度/状态及完整层原Torch对照后保留或回退。原累计10小时不增加，不恢复旧截止。

Self QKV本轮完成：混合函数负结果保留，隔离8warp有限门及完整层正向，最终全部阶段对原Torch重测、两机采用和清理。该检查点之后同形状其余投影已在M39-PROJECTION-FOLLOWUP完成，保留先前阶段身份。

- [x] M39-PROJECTION-FOLLOWUP：获准接续三个同形状投影及一个有依据Self QKV主循环方案；有限门、完整层与新版完整阶段表，原累计10小时。
- [ ] M40-DUAL-UNTIL10：当前算子结束后，两远端卡去重运行既定完整pair；GPU0按预计任务时长在10/23约10:00自然收尾停派，GPU1继续，总160小时不增。

M39本轮完成，最终同轮原Torch完整单层1.2967/1.8469/2.9910×；GPU0已真实交接，临时双卡首GPU0完整pair审计通过。M40-DUAL-UNTIL10已于10:02:27自然收尾，GPU0共150局/50组且owned清理核验完成；GPU1继续，M40总实验仍进行。

- [x] M40-MIDDAY-UNTIL14：用户中午新授权GPU0从新namespace接续未运行完整pair至约14点，GPU1继续；原160小时不增，已14:01:34完成18局/6组并清理，GPU1继续。
- [x] M39-DAG-WEB-PROMPT：固定319ac6a源码、现状/实测边界/用户想法整理，由网页版独立判断方向与优先级；原规定详细设计的交接已被用户纠正，无新算子GPU作业。

- [ ] M40-MICROWAVE-FINAL-DIAG：按用户顺序等待微波炉100组三臂完成后核查Target官方复现与投机额外失败；CPU完成提醒已启动，原队列/资源不变。[有界计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/microwave-completion-20261023/PLAN.md)。

- [x] M39-LOCAL-WARP：执行新授权T0–T4；[计划](../../experiments/local-warp-pipeline-20261023/PLAN.md)，严格数值、逐项完整表、独立A，GPU≤120分钟且旧累计不重置。

M39-LOCAL-WARP闭合：主线不采用，副线L13仅隔离研究采用；四权限false、GPU0清理。本轮附件明确要求D/G强对照，优先于此前仅原Torch的本轮展示限制；原记录不删除。后续新方向待讨论。

M39-RTX5090-PORT已完成有界测量与回传，数值门未通过；保留迁移诊断，未采用为生产实现。当前M39下一步是5090投影数值契约核对，非自动启动新调优。

M39-RTX5090-PRECISION已闭合：三KV有限逐位通过并回传，原迁移失败保留。下一步重测当前代码性能，旧迁移速度不继承。

M39-FFN-TMA-PIPE闭合：三个新候选均有限精度通过但慢于GRAIN，保留原GRAIN；单输出槽放得下，未缩tile；源码/结果返回、owned退出。下一步是根据已保存证据讨论主循环与交接开销，不自动追加候选。

M39-FFN-SPLITK已闭合：仅K段任务化，实际采用SPLITK-M64-FFN2；原GRAIN及TMA负结果保留。下一步建议处理三个小投影，本次未启动。

M39-PROJECTION-SPLITK闭合：有限逐位通过但完整层退化38.09%–42.70%，不采用；已恢复FFN2-SPLITK并重测当前阶段表、public两流。具体原因分量未定量，不自动追加实验。

M39-PROJECTION-TMA-CONTINUOUS闭合：三KV有限逐位/计数通过但完整层慢1.065%–1.384%，恢复FFN2-SPLITK，候选负结果保留。后处理warp重叠未实现，下一步先分量诊断。

M39-PROJECTION-TMA-COMPONENTS闭合：r0细插桩超限，r1修正通过扰动筛查；主循环观察约57微秒、后处理相关约9微秒/tile，不能作为原实现精确stall占比。采用代码未改，先讨论输入交接，再决定输出warp。

M39-TMA-INPUT-HANDOFF的纯八MMA提前交接方案未单独运行；用户新批准的四MMA分工替代该实施路径。M39-PROJECTION-WARP-SPECIALIZED已闭合并研究采用，原A和失败保留。

- [ ] M40-STORAGE-RACE-RESUME：用户要求持续到完整，已修复文件扫描竞态并双卡接续；一次C补齐与CPU完整覆盖watch自动接续，见[计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)。

M39-FFN-HANDOFF闭合：B/C/D/E真实有限逐位均通过但正常整层及同轮FFN均无收益，四者不采用，A六文件byte不变；D初次编译失败保留。33自有身份退出，5090 context为空，双A100继续335有效组。下一实现未启动。
