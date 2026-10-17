# 当前研究恢复入口

更新时间：2026-10-17 13:18，北京时间。派生短入口；原始FAILED/tile/协议保留。先读[索引](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再恢复当前Cycle，不展开全部旧知识。

## 当前终态与权限

**M15六对十二局完整，投入NO_GO，GPU与SSH已清理。** [本轮结果/证据](experiments/tf-pf2-autodl-20261017/RESULT.md)。冻结同一TF_ON，只比较连续C与PF2 P；C3/6、P5/6，rescued2/lost0。风险0/3→2/3，对照3/3→3/3；4/26两臂失败。控制1/36为260→405，长度比1.5577；共同成功三对P50/P90/max=1.0043/1.4470/1.5577，P90/max、风险全成功和控制罚分门失败。风险救回不能抵消。

全体实际命令C3110/P2166，800失败罚分3122/2170；风险罚分2400/1305、对照722/865。失败实际796不补成800。探索性ΣT_round/Σg为.04068/.05251秒（P/C1.2909）；共同成功总policy成本165.90/226.99秒。容器GPU PID无法完整映射，连续成本可比性UNKNOWN，两项成本门UNKNOWN；不是正式E2E加速或原Target对照。

执行源码9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a，科学父4f04。六对物理/controller/RNG/双相机保存组件匹配；首两S现场有效KV/RNG比较通过。reset/S0_before只排除原native覆盖的六个lazy scheduler字段，差异原件保留；不是全部隐藏状态/离线全KV证书。首两S的既有score与depth未清晰区分风险/控制，不能拟合新阈值。

r1配置导入失败、r2两client零环境步基础设施失败均保留；原T0未重置。r3实际runner20.89分钟，从原授权到终态1.962小时；累计server生命周期.728764 GPUh，非CUDA active。1267原件约52.5MiB已回传，自有组EMPTY/guard CLEAN，85登记身份无存活，双GPU各1MiB/0%、计算进程空。**实例尚未关机，用户可关闭。** 临时249MiB资产包和20.1MB证据包双端删，原件保留。

[12h批准](memory/global/requirements/decision-PF2源码执行与AutoDL十二小时窗口.md)从11:11:41起，原22:56停新/23:11清理硬帽未改，但有限面板已结束，不为用余额扩任务。12局/49k/3GiB不扩；16054已计数forward，r2缺COUNTER保守8000、非episode留1000，保守25054/49000。未测全局active时间/非episode真实调用UNKNOWN。

[AutoDL无smoke/独立测试/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)含loader隐式调用；必要审计随完整任务。用户已将[优化reset设为所有后续实验默认](memory/global/requirements/decision-实验默认启用等价reset优化.md)：已验证完整512-token BF16 bank、全臂同工程条件，本轮十二次15–107ms，冷启动/首次构建另列。

四权限始终false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON NO_GO/retention、22收据缺口/FAILED、全局A_off保持；无120/独立TEST/新TF正式45chunk/正式E2E。TF64结构/API暂冻研究，当前连续及PF2配方都未验收。

## 固定快照与前次质量

公开4f04b35e44cbc00483dc6245ad57abec718385c9 / wam-tf-draft-handoff-20261017-paused72永久保持72/96、MLP0、PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED。live77/96另列，不跟main覆盖；历史b0dfac3a不变。[77结果](experiments/tf-effect-closure-20261016/RESULT.md)：fresh T20/20、CONT19/20、ON20/20、MLP1/1（原计划20）；risk T6/8、ON4/8。MLP1/36预算中断非算法失败，余19不自动补。

三条T成功/旧连续TF失败：1/29（264→796）、3/26（240→796）、4/26（249→796）；物理/reset/RNG/双相机保存组件exact，不等于根因已知。本轮新C/P不能更正这些旧失败。risk T/TF实际3244/4371，罚分3252/4387；fresh共同成功长度.990745/1.020326/1.027027。旧70预算FAILED/v2暂停non-budget issue、22局部cleanup缺证保留，三来源PARTIAL_DESCRIPTIVE_ONLY。

## 模型、训练与数据边界

TF64 53070参数：16×17token，state7+previous7+σ/nextσ/t÷1000；单层pre-norm D64/H4/FF256，GELU/drop0、学习位置、center±delta双7维残差头，无图像/prompt cross-attn。MLP96 52928参数仅baseline。根previous用已有target方向，递归previous来自父draft；内部FP32、生产返回BF16方向/FP32打分。

一次新30完整采集9383.734s/23685forward，生成策略旧TF_CONT3000而非当前TF_ON。FIT11416/CAL6805 reached行，11170反事实另留；旧target FIT159691/CAL20237，旧mixed14997=3906 A_off+11091历史TF。child2500同起点seed20260923，三条件固定500更新/b512/lr3e-5/AdamW/wd.01，norm/loss不变，CONT复用。一次机会已用完，不擅自扩训。

真实AST链tf_effect_train→tf_usable_train→finetune.mse_and_gap；主loss回归decoded raw velocity，std只参与decode/gap，无主MSE的std除法/Δσ权重。完整27042行GPU CAL仅adapter/decode、teacher forward0，不证明B2/B14 teacher一致。collector标签是对应query CFG后target方向；video guidance5/action1，外部仍CFG2。新FIT无足够逐cache obs/完整KV，227维表不能补造视觉条件并在新latent重询teacher。

PF2严格S S T，成功policy返回才计数，reset清零/cache不计，S全拒绝仍算S；T canonical CFG2 action50更新/51forward含terminal。不是bonus或环境回滚。τ=.007839680526450805、absΔσ×L2、prefix_then_cost、K2/B2/bonus不变；7状态×CFG2=B14，每attention仍Q16，private KV不默认共享。固定六pair/顺序/eval seed20260922及全量原件见本轮PLAN/RESULT，不加Target/MLP臂。

## 系统与下一步

因果主线：投机重组→可调度候选→超过最佳B14→质量约束下有效成本。H1真实shape/H2批处理有证据，H3有限pilot、H4未成立。30层D3072/H24×128/FF14336；Linear M32/96/224已合并，attention仍Q16。G1→G2 L2请求−32.81%、DRAM仅−1.31%、occupancy降，不能写HBM减半。旧tile/投机/reset倍数不相乘冒充新收益，原tile只读、不恢复撤回融合。

下一步建议另登记teacher路径诊断，区分B2/B14差异与draft提交闭环影响；**尚未启动/新增批准**。不自动训练、调tau、PF3/4、补旧19、120、TEST或正式性能。双卡可并行完整pair，不能让小draft训练获同等收益；正式性能仍本机独占且原质量放行。

旧系列墙钟11.0265h、已知85039/保守97039forward留历史，不借旧余额。项目两旧下载残片已回收2.38GiB（du137.25→134.87GiB），canonical/raw/KV/失败保留。HW固定v15/legacy，不升级/全量sync；保护state/cycle/rules、结构化规则和491原tile记录metadata未变。

10/18作者profile检查仅记录；10/22 go/no-go；10/23冻结允许的候选/API/claim；10/28主表；**11/01 18:00中文证据包交老师**。用户整理飞书，不新建本地论文稿；证据不足收缩claim。CVPR2027官方日期临近再核，未确认独立abstract规则，不套往年。
