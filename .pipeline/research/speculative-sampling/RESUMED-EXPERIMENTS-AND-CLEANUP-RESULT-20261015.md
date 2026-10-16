# 恢复后实验、质量边界与空间清理

更新时间：2026-10-15 22:02，北京时间。本机30局及同seed parent10均已完整终态，所有自有GPU进程清理；AutoDL原100继续，新接续已实际排队并只读等待旧100。用户已恢复本机GPU，允许多核CPU；不改tile Transformer，不恢复撤回的融合，不运行smoke或独立大文件SHA作业。受保护state/cycle/rules未改。

## 当前结论

本轮已完成reset提速等价验证、历史179928条真实监督审核、TF/MLP各三seed预训练及两阶段训练、新T/MLP/TF30条完整LIBERO Long开发任务、同seed TF23原parent10条完整消融、失败产物清理。TF小模型可以稳定运行并减少target调用；两阶段TF比同seed parent总体800失败罚分更低，但没有通过成功数工程门，不能称投机侧质量已经收尾。

结构可固定为下一轮研究骨架：TF64，16个action token，1层pre-norm TransformerEncoder，4 heads，FFN256/GELU，dropout0，53070参数；输入每token为state7、previous target velocity7、σ/nextσ/t三标量，输出两组7维方向残差。没有cross attention或图像/prompt输入。MLP96为52928参数公平对照。结构固定不等于训练配方和接受标准已经获质量批准。

本轮没有MegaTransformer算子收益，没有新TF正式chunk/task/E2E加速比。先前撤回的融合与tile代码不动；扩大质量时可以并行设计算子与写方法，但最终速度表仍需稳定候选和本机独占计时。

## 新30条完整任务的实际结果

来源：`投机采样/results/runtime/p0-target-trace-draft-fresh-dev-v1-20261015-r2/`。新T、微调MLP96、微调TF64分别10条Long任务，task0–9/init24，原800 horizon；20:04开始，21:13完成并清理全部自有进程，总target调用31076。两draft分别完整20237条真实CAL的同GPU adapter/原decode batch1逐位核验通过。无截断或事后重试。

| 方案 | 成功 | 实际环境步总数 | 全部局步数P50/P90/max | 800失败罚分均值 | target调用/chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| 新T | 10/10 | 2721 | 261.5/353.8/379 | 272.1 | 13152/178 = 73.888 |
| 两阶段MLP96 seed22 | 9/10 | 3316 | 258.0/575.5/796 | 332.0 | 9225/213 = 43.310 |
| 两阶段TF64 seed23 | 9/10 | 3108 | 256.0/407.2/796 | 311.2 | 8699/200 = 43.495 |

两draft都在task7跑满50 chunks后失败，T在333步成功。task7初始物理状态、环境/target RNG和两相机均逐位相同，不能把此负结果归因于task8相机差异。TF的task8成功364步，MLP551步，T379步，但此task的跨臂首帧像素不完全相同，不能作完全相同观察的强因果比较。

TF与T双方成功的9局，实际步数比P50=0.9956、P90=1.0415、max=1.1622，已满足此前1.10/1.20/1.30长度工程条件；仅按初始图像也逐位的8个双方成功task，P50=0.9978、P90=1.0566、max同为1.1622。长度条件不能遮掉第10局失败，故总体质量门仍不通过。MLP双方成功9局的最大比1.4538来自task8，另8个像素严格双方成功task最大1.0336；必须同时标明观察配对差异，不能任选更好子集替代全体统计。

原作业严格配对门因此仍为FAILED：task8腕相机T–MLP有5像素/15通道不同，T–TF有4像素/12通道不同，幅值最大2/255；其他9任务两相机均逐位相同。独立CPU终态诊断以实际保存的sim/controller数组、环境RNG、target CPU/CUDA RNG、RESET状态证明全部物理/RNG配对成立，并分别报告像素严格子集。没有放宽原门、注入图像或制造原RESULT。

独立诊断：`投机采样/results/runtime/p0-fresh-dev-terminal-diagnostic-v1-20261015-r2/{DIAGNOSTIC.json,SUMMARY.md}`。7406棵已记录接受树的prefix/cost/τ/bonus规则审计无违规；未保存的原始target/candidate方向不能在该审计中重建。本诊断首个CPU尝试发现reference/candidate哈希标签写反而拒绝；仅修新工具标签并重新独立分析，原r1拒绝和原实验FAILED均保留，原像素数/数值/门未改。

## 训练进展与同seed消融

历史TRAIN194完整轨迹共179928行监督，FIT174为159691行、CAL20为20237行；另CAL50的49245行只审核、不加入FIT。数据由真实target原query逐tensor核验，不能把离线样本数当成功率证据。

预训练TF/MLP各三seed、2000更新，CPU936.42秒；所选CAL中位seed22的局部通过率95.04%/95.41%。这轮原40完整任务因task8相机配对FAILED，描述性成功数T/旧A_off/预训练MLP/预训练TF为10/8/10/9；TF task3严格同初态失败有效保留。

两阶段训练为原2000更新再500更新，batch512、AdamW lr3e-5、wd0.01、target/mixed各半、原normalization不重拟合。六组CPU587.61秒完成，所选TF23/MLP22局部CAL95.0593%/95.4567%；两者闭环实际9/10，说明微小离线提升不能替代成功率测试。mixed14997行中3906来自旧A_off、11091来自历史研究TF，均非本轮新TF on-policy数据。

追加唯一对照已完成：TF23真正2000步parent，沿child metadata的原checkpoint路径与字节绑定，53070参数、同seed、normalization逐位相同。独立源码/CPU实加载/协议审阅通过后，21:26:57启动，完整10局收集用时1342.93秒，21:49已分析/清理终态，rootsession exit0，实查nvidia-smi无compute进程。固定相同τ/K2B2/selector与target RNG，完整20237同GPU CAL逐位通过；一个model和一个EGL client、CPU2，无额外训练/阈值/seed调整。

| 同seed TF23对照 | 成功 | 失败task | 实际步总数 | 800失败罚分均值 | T双方成功步数比P50/P90/max |
| --- | ---: | --- | ---: | ---: | --- |
| parent2000 | 9/10 | 1 | 3328 | 333.2 | 0.9974/1.1481/1.4715 |
| child2500 | 9/10 | 7 | 3108 | 311.2 | 0.9956/1.0415/1.1622 |

parent与child所有10个task的物理/RNG及两相机初始像素均逐位相同；它们相对T都是9个相机严格task，task8除外。两阶段训练将task1从parent796步失败改善到child251步成功，同时将task7的parent490步成功变为child796步失败。成功数持平，800罚分均值下降22（6.60%）；不能称微调全面退化或成功率已保证。parent超过1.30最大长度条件，child长度条件通过但成功数9<10。父子分析7478棵记录树无规则违规（其中含复用child的3611棵，不能与fresh30简单相加）。

真实证据：`投机采样/results/runtime/p0-target-trace-parent-ablation-v1-20261015-r1/{RESULT.json,REPORT.md,PARENT-REFERENCE-PAIRING.json,FINAL-OWNED-CLEANUP.json}`。父子完整对照不是新的独立TEST，不复用旧TF22。

后续父/子TF的探索性扩大评估，在运行parent前已登记：成功数更多→全10局800失败罚分均值更小→并列选parent。实际规则因此选择child2500/TF23，checkpoint保持原字节`9cb92a4862579edc9cfb99210b0535e620daba56dd5abcc094a7d735dbadab07`。实际CPU local_gate已核验；`quality_gate_passed=false`，唯一失败工程项为成功数不低于T（9<10）。这是供扩大风险测量的研究选择，不是最终合格/全局部署；原v2和正式性能入口的正向门不改。见[统计质量协议](STATISTICAL-QUALITY-PROTOCOL-20261015.md)。

## reset和已有速度证据

Long10十任务加空负提示的原T5完整512 token BF16最终embedding已缓存；CPU7首次构建706.01秒。裁padding尝试0/11逐位相同，未采用。原编码/缓存的T/A_off×task0/1共8个完整任务、4对对照通过；全部418件保存文件、2254组tensor/array值/形状/类型及动作、环境、target RNG、30层有效KV和VAE等价。

原reset内部127.8–154.2秒→24.8–27.5毫秒，client reset RPC约103–128毫秒。这是共享GPU条件下功能等价和耗时诊断，不能当新TF或缓存后的正式E2E收益。旧AutoDL100已经启动，未中途修改其慢reset实现。

保留的正式旧A_off完整chunk短/中/长KV为1.601926×/1.544039×/1.500061×，45个完整chunk、有实际输入及全部输出/30层有效KV/路径/RNG逐位验证。旧四任务双方4/4成功，含reset/请求/环境动作的自然episode为1033.403/898.071秒=1.150692×；含model启动到client退出为1.090947×。原reset占46.81%、policy44.20%、环境动作4.16%，说明旧E2E主要被reset和模型推理限制，非仿真动作独占。

新TF45完整chunk源码已准备但本轮新质量门不通过，未运行或绕过。不能将43.495 target调用/chunk的调用量降低直接写作时延加速。

## AutoDL与时间边界

原AutoDL任务是T/A_off的50对、100完整质量任务，不是TF/MLP/PF2选择训练。22:01:53实际只读进度78/100、无FAILED；原任务不重启、不修改。纯归档watcher每60秒检查实际终态和清理，完成后先保存小型结果。按当前约6.5分钟/局，预计10/16凌晨00:30–01:30旧100收尾；失败跑满horizon或资源波动会延后，不把估时当完成证据。

新research quality链源码及增量独审已通过，实际CPU最小bundle已生成，70,327,498字节、260份来源（含实际native二进制），完整20237 CAL逐列拷贝与原数据相同，无大模型权重/原始训练采样搬运。root已复核实际研究选择、原checkpoint metadata不改、冻结来源与独审记录。唯一dispatcher-r2于22:01:50启动，PID2222687，22:01:53实际状态WAIT/old100_incomplete；尚未部署远端新包或启动其CPU/GPU。旧纯归档watcher保留为备份。dispatcher-r1在0SSH/0GPU前因本机Python包搜索路径报错退出，仅小日志；两行入口修复已以真实direct-CLI环境独审通过，r1失败及审阅增量保留，未重打包或修改25份远端源码。

接续顺序：真实父10/新30终态→预登记TF父子研究选择→旧100实际终态归档→远端构建11条bank→4完整reset等价→新的T50/TF50。不能复用旧T50，因为旧100没有完整保存本协议需要的target CPU/CUDA初始RNG证明。真实20237 CPU和同GPU接口核验、4完整reset和单一GPU worker仍须实际通过；源码/导入审阅不冒充这些实际数值门。一次累计4小时从远端CPU bank开始计，含setup/等待/分析并留300秒清理，最迟10/16 10:30退出，无自动续租。若旧100按预估完成，新链预计04:30–05:30左右终态；但本机新30推算单100纯episode已约3.34小时，加bank/4reset/启动可能超过4小时，因此不承诺104条必能完成，预算不足保留PARTIAL/FAILED与未完成清单，不缩horizon或冒称完整质量。AutoDL只做算法质量，不作正式论文性能；脚本退出并不代替用户在租卡控制台关闭实例。

准备/队列证据：`投机采样/results/runtime/p0-autodl-research-quality-preparation-v1-20261015-r1/{ROOT-BUNDLE-REVIEW.json,ROOT-DISPATCH-LAUNCH-r2.json}`、`p0-autodl-research-quality-dispatch-v1-20261015-r2/status.json`。源码：[research quality](../../../投机采样/experiments/autodl_research_quality_v1/README.md)及[dispatcher](../../../review/OMISSIONS.md)。最终审阅含`INCREMENTAL-REVIEW.json`和`DISPATCH-NAMESPACE-INCREMENTAL-REVIEW.json`，均在`p0-autodl-research-quality-independent-review-v1-20261015-r1/`。

21:42只读核查远端`cpu.max=1000000 100000`，实际CPU配额10核（在线80核、cpuset0–79不等于可独占80核）。用户已允许多核且优先快结果，因此仅新research版本计划bank与target server用7核、EGL client2核、本机准备/完整CPU CAL仍2核；旧100仍CPU2，不中途改。原CPU2逐条T5约233秒、11条可能约42分钟，旧复制的bank1800秒上限有超时风险；新bank单步上限3600秒，仍受共同4小时及绝对截止的更早帽。7核不是已实测收益；完整原helper/512BF16保持，4条完整reset逐位等价门仍须真实通过。配额原始只读记录见`evidence/AUTODL-CPU-QUOTA-READONLY-20261015.json`。

## 已实际清理

本次确认删除4份旧失败张量615922780字节，587.39MiB；累计清理约23.81GiB。保留失败原因/日志/摘要/配置/清理记录，删除目录注明原张量已不可独立复算。清单和是否仍需保留的失败目录见[清理报告](FAILED-ARTIFACT-CLEANUP-20261015.md)。原40和新30的完整负结果、reset有效数据、20份真实计时fixture、原trace/权重、tile代码和HW均保留。

22:00项目实际分配138545782784字节，约129.0GiB，包含新30、parent10及接续包；后者checkpoint/CAL预测使用不可变硬链接减少本机重复存储。整卷剩余空间受其他用户影响，不作为本次释放量。失败marker不等于全部数据垃圾，尤其不能按目录名批量删除后来CPU恢复的有效reset结果。

## 下一步与截稿倒排

1. 同seed parent10已完成，保留TF64结构，以child2500作为扩大质量的研究候选；最终质量合格条件未过，不能声称训练配方/接受器已收尾。
2. research质量链已独审/最小化打包并实际排队，等旧100结束先汇总后接续；新T50/TF50用于风险量化，必须如实保留负结果和预算未完成状态。若成功率差距持续，优先收集新TF在FIT初态真正访问到的query及其target方向，完成有限on-policy聚合训练/保守接受器对照，再做新的完整DEV；旧14997混合FIT不能代替新TF on-policy，也不能按已见失败task硬编码退回T。预计额外数据/训练/完整DEV需约1–2天，取决于GPU窗口，不能给成功保证。
3. 算子可先基于冻结K2/B2 verifier接口、真实输入和原30层数值参考做设计/profile，候选质量和完整chunk正式计时分开。用户考虑好算子方案后再开始新融合，不恢复旧撤回实现。
4. 本机独占窗口再补最终候选的短中长KV完整chunk、成功task及含reset/cold-start E2E；Long10为优先，第二模拟器RoboTwin须官方可用权重和时间允许，暂不同时铺LIBERO100/RoboDojo/真机/多骨干。

CVPR2027官方日期页与CFP：注册Nov10 AoE=北京时间11/11 19:59，正文Nov16 AoE=11/17 19:59，补充Nov23 AoE=11/24 19:59；官方当前未单列独立摘要截止，内部按注册前锁摘要/标题/作者。给老师完整中文稿目标11/01 18:00；OpenReview作者资料10/18前初检、最迟10/25，非机构邮箱新profile可能需两周。10/18–22扩大质量与算子设计并行，10/22范围决策，10/23–27算子/必要第二模拟器，10/28–31冻结主表与中文稿。稿件正文逐段在会话给用户，不生成本地初稿。

官方来源：[Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)、[Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)。文献训练/接受器依据见[训练与质量证据](TRAINING-AND-QUALITY-EVIDENCE-20261015.md)、[CCF-A接受标准调研](ACCEPTANCE-CRITERIA-CCFA-20261014.md)、[官方配对标准](EVALUATION-PAIRING-STANDARDS-20261015.md)。
