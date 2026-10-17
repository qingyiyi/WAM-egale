# 当前研究恢复入口

更新时间：2026-10-17 18:42，北京。派生短状态；先读[INDEX](INDEX.md)、[上下文](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再读当前Cycle四文件；原始FAILED/tile/协议不改。

## 当前M17：三套独立TF训练与完整三臂

用户明确批准Spatial/Object/Goal分别用对应社区Target、同一TF64训练方法，比较Target/连续TF/同权重TF+PF2的成功率、trace长度和E2E成本。M16 Long继续；90尚无可信同代Target，未启动，不将本次当取消90。三个Wjjjh Target非官方，发布者SR不继承。[完整PLAN](experiments/libero-four-suite-20261017/PLAN.md)、[实际作业](experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)。

Object/Goal远端各10,177,831,668B权重完整，本机Object完整副本也已就绪。Goal/Spatial完整512-token BF16 bank完成（CPU2455秒），Object远端也完成（CPU1220.5秒）；原本机bank超时、传输超时和准备错误保留。新共享入口8eaf130继承9aa数学/固定训练，三个远端controller32543/31852/31853等待原Long清理/必要资产，新套件模型/episode/训练仍0。旧准备控制器退出不改原记录。

固定训练：Target200=FIT180(init0–17)/CAL20(18–19)；PARENT2000采30(init20/21 FIT、22 CAL)，target/该FIT 50:50训CHILD2500与CONT3000；CONT采30(init23/24 FIT、25 CAL)，ON从同CHILD+500，不是CONT续到3500。仅final/seed20260923、不搜结构/seed/tau。所有自然失败保留，只实际reached teacher标签训练，反事实另列；没有完整条件则不能新latent重新query教师。

每套最终500case×T/C/P=1500局，Target前200只读复用；另训练采集60局，总1560/套、4680新局。为本机减负，零新增数据前新登记Object/Goal全套固定AutoDL GPU0、Spatial固定GPU1；同卡原子双租约限制模型/CPU；最终case0–249、250–499两shard，可显存许可独立双worker。完整50init含FIT/CAL/DEV，训练未见26–49子集另报，四权限不变，不签独立TEST。

M17 master10/17 17:11:31起，10/21 16:56停新/17:11清理；18,723,000尝试、研究证据15GiB、资产本地40GiB。每局1200秒/4000尝试/horizon800，整局预留不足不启动，失败/等待/准备/清理计账，无自动重试。现有AutoDL可用至已批准工作完成，不采购新卡；source/local/remote控制器与终态以job记录为准。粗估新增48–72h，实际首批任务后更新，96h是止损帽。

## M16 Long仍运行，原账本独立

原v1 23组三臂69完整局/72593forward只读接续；v2 d5448bb runner28232/guard28231四worker、每卡2模型/环境/KV/RNG、总16CPU。18:42新285闭合局＋原69＝354，另有运行中局，未全量。原48h不重置：10/19 13:40:56停工、13:55清理，6,001,000forward/10GiB含fixture3GiB/free5GiB。[原索引](experiments/libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)。

自然失败不提前停止完整描述覆盖；初态/首两S差异保留FAILED/严格pair INVALID，API/数学/资源硬停。v1 capture重复字段、v2有效KV桶缺口保留UNKNOWN，尚无合格新TF全状态replay。不能将compact训练表或保存latent/动作称全KV/视觉条件fixture。并行质量的时间仅探索性成本，正式replay只支持其边界、不能代替完整机器人E2E。全部实际命令步数与800失败罚分分列，更快失败不是加速。

## 永久保留的旧质量与模型身份

四权限始终false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON/PF2 NO_GO、retention、22清理缺证、所有FAILED和全局A_off不变；无120、独立TEST、新TF正式45chunk或正式E2E。

固定4f04b35 paused72永久72/96、MLP0/PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；本机live77另列，main/b0不替代。旧ON fresh20/20不覆盖risk4/8对T6/8；三条T成功/TF败(1,29)/(3,26)/(4,26)保存匹配组件，根因未定。[原77结果](experiments/tf-effect-closure-20261016/RESULT.md)。唯一Long新训练机会已用；新增三套由用户另批准，不借此继续Long tuning。

M15六pair完整C3/6、P5/6，救回2/lost0，4/26两败；控制1/36 260→405。共同成功长度P50/P90/max=1.0043/1.4470/1.5577，投入NO_GO。探索ΣT_round/Σg .04068/.05251秒、共同成功policy165.90/226.99秒，成本门UNKNOWN，非正式E2E。旧全部源/收据/清理原件保留，[五报告](experiments/tf-pf2-autodl-20261017/RESULT.md)。

TF64=53070参数，16token×17feature(state7/previous7/σ/nextσ/t÷1000)，1层pre-norm D64/H4/FF256、GELU/drop0、学习位置、双7维残差头，无图像/prompt cross-attn。MLP96=52928仅baseline。主loss回归decoded raw velocity，std用于decode/gap；27042行GPU CAL是adapter接口，不证明B2/B14 teacher一致。τ=.007839680526450805、absΔσ×L2、prefix_then_cost/K2B2/bonus；生产BF16方向/FP32打分。PF2=SST完整policy请求计数，reset清零/cache不计，全拒绝S仍计；T canonical50 scheduler更新/51 action调用含terminal，不是bonus或物理恢复。7候选×CFG2=B14，每attention仍Q16，private KV不默认共享。

## 研究与资源边界

系统主线：投机改变小action工作负载→候选可调度→超过最强B14→质量约束有效成本；H1/H2有证据，H3有限pilot、H4未立。D3072/H24×128/FF14336，Linear M32/96/224已落库，attentionQ16；G1→G2 L2−32.81%、DRAM−1.31%、occupancy降，不能写HBM减半。原tile不动，不恢复撤回融合，旧倍数不乘。

AutoDL[无smoke/独立测试/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)包含隐式loader，沿源码完整任务inline审计。全部臂[默认优化reset](memory/global/requirements/decision-实验默认启用等价reset优化.md)，原512-token BF16，冷启动/首次bank另列。M15 reset15–107ms不冒充新套件结果。本机他人GPU在用，新增重计算优先AutoDL，不杀别人/改系统或时钟。本轮用户明确暂不扩容：37f15d1调度PID32254已启动等待Long全量及实际清理，届时才核查轮换其远端Transformer缓存给Spatial，T5/VAE/tokenizer、本机原件与全部结果保留。本机本次清7个失败下载残片0.917GiB，项目144.52GiB、余约94.9GiB。Object新增结果已改远端OS盘（5GiB套件帽/空闲线5GiB），不占数据盘；Long原始结果准备完整归档到本机后才退远端重复bulk/COUNTER，归档128d435/PID443492已启动等待，当前复制0/删除0，不删唯一证据。新96h不覆盖旧job deadline；实例关机与进程退出分列。

HW固定v15/legacy，不升级/旧全量sync；保护state/cycle/rules/结构规则/原tile记录不动。Cycle旧四文已逐字归档，当前入口语义压缩，[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)引用历史和原始结果，不以截前80行当摘要。磁盘清理和文档精简分开；旧2.38GiB清理不当本轮回收量。

10/18作者profile只记录；10/22 go/no-go；10/23冻结candidate/API/claim；10/28主表；11/01 18:00中文证据包。中文稿逐段会话交飞书，不建本地论文正文。CVPR注册/正文/abstract规则执行时按官方核验，不沿旧年份假设。
