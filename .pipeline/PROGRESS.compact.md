# 当前研究恢复入口

更新时间：2026-10-17 22:08，北京。派生短状态；先读[INDEX](INDEX.md)、[上下文](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再读当前Cycle四文件；原始FAILED/tile/协议不改。

## 当前M18：阶段备份与换机；M16/M17待接续

旧AutoDL已自然收尾189组三臂/567局、0partial，全部37349文件/1.84GiB原件备份完成/CLEAN、旧源0删除。用户最新允许新机接入前继续：2b1f039独立namespace接续311未开始组，当前4模型READY、完整任务运行中，按完整30局边界可暂停；新SSH未收到、新结果需增量备份，当前不能关旧机。[阶段结果与迁移](experiments/autodl-stage-migration-20261017/STAGE189-RESULT.md)。三套独立TF研究仍保留，90缺可信同代Target未启动；Wjjjh Target非官方，发布者SR不继承。[完整PLAN](experiments/libero-four-suite-20261017/PLAN.md)、[实际作业](experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)。

Object/Goal远端各10,177,831,668B权重完整，本机Object完整副本也已就绪。Goal/Spatial完整512-token BF16 bank完成（CPU2455秒），Object远端也完成（CPU1220.5秒）；原本机bank超时、传输超时和准备错误保留。新入口继承9aa固定数学/训练。最新空闲本机授权下，Object source652e83c已实际READY（setup749.6s），首局1200s超时，20:24停止/CLEAN，0完整局；远端Object32543零工作退役，旧终态保留。Goal/Spatial已按换机请求PAUSED_PARTIAL、租约已释放；尚无新增闭合episode或训练。

固定训练：Target200=FIT180(init0–17)/CAL20(18–19)；PARENT2000采30(init20/21 FIT、22 CAL)，target/该FIT 50:50训CHILD2500与CONT3000；CONT采30(init23/24 FIT、25 CAL)，ON从同CHILD+500，不是CONT续到3500。仅final/seed20260923、不搜结构/seed/tau。所有自然失败保留，只实际reached teacher标签训练，反事实另列；没有完整条件则不能新latent重新query教师。

每套最终500case×T/C/P=1500局，Target前200只读复用；另训练采集60局，总1560/套、4680新局。最新用户允许空闲本机GPU分担，零新增数据前Object全套固定本机SXM4单worker/nice10，两最终shard顺序；Goal固定AutoDL GPU0、Spatial GPU1，原租约/限额不变。同case所有臂同物理卡，不混硬件计时。完整50init含FIT/CAL/DEV，训练未见26–49子集另报，四权限不变，不签独立TEST。

M17 master10/17 17:11:31起，10/21 16:56停新/17:11清理；18,723,000尝试、研究证据15GiB、资产本地40GiB。每局1200秒/4000尝试/horizon800，整局预留不足不启动，失败/等待/准备/清理计账，无自动重试。旧AutoDL现按新决定阶段暂停待换机，不采购新卡；source/local/remote控制器与终态以job记录为准。粗估新增48–72h，实际首批任务后更新，96h是止损帽。

## M16 Long新namespace接续，原账本独立

原v1 23组三臂69局/72593forward只读复用；原v2 d5448bb阶段189组/567局已清理，311组未开始，原PARTIAL保留。新2b1f039 supervisor69459/runner69462/guard69461已启动（READY、首分片运行中），4worker/每卡2模型、server/client总16CPU；保留所有多代源映射，不重复旧组，完整10case/30局屏障可暂停。累计旧attempt debit591787，原48h不重置：10/19 13:40停工/13:55清理，6001000forward/10GiB含fixture3GiB/free5GiB。[原索引](experiments/libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)。

189组仅描述：T183/189、C185/189、P181/189；失败800罚分均值284.81/274.76/284.22。C/P相对T共同成功max长度比1.7004/1.5465，组件FAILED15/27保留；含训练暴露，不签质量通过、独立TEST或正式E2E。

自然失败不提前停止完整描述覆盖；初态/首两S差异保留FAILED/严格pair INVALID，API/数学/资源硬停。v1 capture重复字段、v2有效KV桶缺口保留UNKNOWN，尚无合格新TF全状态replay。不能将compact训练表或保存latent/动作称全KV/视觉条件fixture。并行质量的时间仅探索性成本，正式replay只支持其边界、不能代替完整机器人E2E。全部实际命令步数与800失败罚分分列，更快失败不是加速。

## 永久保留的旧质量与模型身份

四权限始终false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON/PF2 NO_GO、retention、22清理缺证、所有FAILED和全局A_off不变；无120、独立TEST、新TF正式45chunk或正式E2E。

固定4f04b35 paused72永久72/96、MLP0/PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；本机live77另列，main/b0不替代。旧ON fresh20/20不覆盖risk4/8对T6/8；三条T成功/TF败(1,29)/(3,26)/(4,26)保存匹配组件，根因未定。[原77结果](experiments/tf-effect-closure-20261016/RESULT.md)。唯一Long新训练机会已用；新增三套由用户另批准，不借此继续Long tuning。

M15六pair完整C3/6、P5/6，救回2/lost0，4/26两败；控制1/36 260→405。共同成功长度P50/P90/max=1.0043/1.4470/1.5577，投入NO_GO。探索ΣT_round/Σg .04068/.05251秒、共同成功policy165.90/226.99秒，成本门UNKNOWN，非正式E2E。旧全部源/收据/清理原件保留，[五报告](experiments/tf-pf2-autodl-20261017/RESULT.md)。

TF64=53070参数，16token×17feature(state7/previous7/σ/nextσ/t÷1000)，1层pre-norm D64/H4/FF256、GELU/drop0、学习位置、双7维残差头，无图像/prompt cross-attn。MLP96=52928仅baseline。主loss回归decoded raw velocity，std用于decode/gap；27042行GPU CAL是adapter接口，不证明B2/B14 teacher一致。τ=.007839680526450805、absΔσ×L2、prefix_then_cost/K2B2/bonus；生产BF16方向/FP32打分。PF2=SST完整policy请求计数，reset清零/cache不计，全拒绝S仍计；T canonical50 scheduler更新/51 action调用含terminal，不是bonus或物理恢复。7候选×CFG2=B14，每attention仍Q16，private KV不默认共享。

## 研究与资源边界

系统主线：投机改变小action工作负载→候选可调度→超过最强B14→质量约束有效成本；H1/H2有证据，H3有限pilot、H4未立。D3072/H24×128/FF14336，Linear M32/96/224已落库，attentionQ16；G1→G2 L2−32.81%、DRAM−1.31%、occupancy降，不能写HBM减半。原tile不动，不恢复撤回融合，旧倍数不乘。

AutoDL[无smoke/独立测试/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)，含隐式loader；全臂[优化reset](memory/global/requirements/decision-实验默认启用等价reset优化.md)，512-token BF16，冷启动/bank另列。原tile不动，不杀他人/改服务时钟。本机CPU16核被root gem5/QEMU竞争占满，Object首局1200s超时/CLEAN，0完整局、不自动重试；空闲本机分担授权保留。37f15d1轮换已停、0删除；T5/VAE/tokenizer、本机原件、所有失败保留。本机本轮仅删7个失败下载残片0.917GiB，项目原144.52GiB、余94.9GiB（备份新增另计）。M18 v1回传慢实际SIGINT退出130/SSH闭合，原FAILED与部分副本保留；v2 635195单tar650.7s成功/CLEAN、37349文件/1970600995B，全部entry/字节/stat核对，无SHA/旧源删除。新GPU结果不在该冻结归档，换机前另增量保存；实例关机和进程退出分列。

HW固定v15/legacy，不升级/旧全量sync；保护state/cycle/rules/结构规则/原tile记录不动。Cycle旧四文已逐字归档，当前入口语义压缩，[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)引用历史和原始结果，不以截前80行当摘要。磁盘清理和文档精简分开；旧2.38GiB清理不当本轮回收量。

10/18作者profile只记录；10/22 go/no-go；10/23冻结candidate/API/claim；10/28主表；11/01 18:00中文证据包。中文稿逐段会话交飞书，不建本地论文正文。CVPR注册/正文/abstract规则执行时按官方核验，不沿旧年份假设。
