# 当前研究恢复入口

更新时间：2026-10-18T17:29:20+08:00 北京。先读[INDEX](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再聚焦当前Cycle四文；不全量展开历史。原FAILED/原tile/保护state-cycle-rules保留。

## 首轮Long已收齐，当前M17仍运行

[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)；[最终汇总](experiments/long-morning-20261018/FINAL-SUMMARY.md)；[最终原件入口](experiments/long-morning-20261018/FINAL-RESULT.md)。11:36:19冻结500组（10任务×50init）、T/C/P各500、1500完整局，本机已回传2668079B。晨报438/旧397原件保留，不混分母。

| 方法 | 成功/500 | SR | 失败 | 实际命令 | 800失败罚分均值 |
|---|---:|---:|---:|---:|---:|
| Target | 483 | 96.6% | 17 | 142861 | 285.858 |
| 连续TF | 484 | 96.8% | 16 | 141230 | 282.588 |
| TF PF2 | 473 | 94.6% | 27 | 146527 | 293.270 |

T→C lost11/rescued12、T→P lost19/rescued9、C→P lost20/rescued9。共同成功C472/P464；成功长度比P50/P90/max：C/T=1/1.073561/2.329167，P/T=1/1.054054/3.223577，>1.30分别6/472、7/464。保留全部失败，失败actual796和罚分800分开；不能用成功子集隐藏失败。

C36/P66首两S严格配对FAILED，Target组件500PASS，原FAILED未改。50init含FIT/CAL/DEV暴露，不是独立TEST。仅首轮完成，后两轮seed/预算未冻结，不声称三轮均值或1500独立初态。

新机正常273组并发暖episode探索耗时比T/C=1.4974、T/P=1.2330；双方成功256/251对为1.4814/1.2809。按同GPU/执行根列源，不混PCIe/SXM；排除模型/环境构建，不是正式独占E2E。正式E2E UNKNOWN。有效action成本仍ΣT_round/Σg，不能用调用数代替。

## 当前资源与队列

Long旧219+正常273+获准补缺8=500。正常分片原PARTIAL_OR_FAILED_LONG500_COVERAGE保留；最终联合COMPLETE只指覆盖，不是quality。主/补缺worker owned groups均空、guard CLEAN，11:37新机两GPU compute曾空；12:14已接续Goal/Spatial，不再空闲。Long reporter/returner/watch真实退出，未杀他人。

17:25三Target池均200完整/CLEAN。Goal/Spatial均最终双worker满载，三臂齐32/38组，唯一完整276/290局；当前共同case T/C/P成功29/29/28和35/35/35，仅部分描述。Object父模型30已收齐、CHILD已完成、CPU CONT250/500（17:30），本机GPU当前空，后续CONT→第二30→ON自动接续。两资源队列仍等原ON，未启动Object final。[本次统计](experiments/libero-four-suite-20261017/PROGRESS-20261018-1725.md)。原预算/四false不变，近ETA每3分钟，其余休息；90仍BLOCKED。

M18迁移完成：全部原件/匹配双环境与四套Target落新机。全部新机工作限~/cxz(实际/root/cxz)，不改系统/复制私钥/建/rjs alias。双SXM4/CPU quota45/RAM180GiB，共享约13TiB不是个人quota或持久性承诺，资产150GiB帽足够四套，无需旧50GB轮换。旧AutoDL00:41 owned/compute空并通知可关闭，不再SSH，实例关机UNKNOWN。

## 原预算和科学身份不变

M16原10/17 13:55:56起，10/19 13:40:56停新/13:55:56清理；6001000尝试、10GiB含fixture3GiB。实际1593717，剩4407283；等待/失败/准备/清理计生命周期，不换起点，CUDA active小时UNKNOWN。

M17原10/17 17:11:31起，10/21 16:56停新/17:11清理；三套4680局/18723000尝试/15GiB。Object phase16:45停新/17:00清理、801000含旧296；指定case0/1已获准补齐，其他失败不自动重试。

三小套件各自Target200=FIT180(init0–17)/CAL20(18–19)，PARENT2000→30局(init20/21 FIT、22CAL)→CHILD2500/CONT3000→CONT30局(init23/24FIT、25CAL)→同CHILD+500得到ON3000；不是CONT3500。只final、seed20260923，不能借Long权重或发布者SR。全50init有暴露，26–49另报也不自动签独立TEST。

四项false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON/PF2 NO_GO、retention、22清理缺证、4f04 paused72/96/MLP0、本机live77、全局A_off保持。唯一Long新训练机会已用，不自动120/TEST/新TF正式45chunk/后两轮或新结构搜索。

TF64=53070参数、16token×17feature(state7/previous7/σ/nextσ/t÷1000)，1层pre-norm D64/H4/FF256、GELU/drop0、学习位置、双7残差头，无图像/prompt cross-attn；MLP仅等容量对照。TF_ON3000/normalizer/tau=.007839680526450805、absΔσ×L2、K2B2/bonus/prefix_then_cost、BF16方向/FP32打分固定。PF2=SST完整policy请求，reset清零/cache不计，T50更新/51action调用含末尾；不是bonus或物理回滚。

## 系统、证据、清理和日期

当前共享Transformer约5.09B(5089465566元素，不含T5/VAE)，30层/D3072/H24/FF14336，不能按论文双流0.35B推action耗时。7候选CFG2=B14，attention仍Q16/private KV不默认共享；Linear M32/96/224已有库合行。H1/H2有证据、H3有限pilot、H4未立。G1→G2 L2−32.81%、DRAM−1.31%、occupancy下降，不声称HBM减半。原tile负结果保留，不恢复撤回融合/乘旧倍数。

[计时与重放准备](experiments/long-morning-20261018/PERFORMANCE-BOUNDARIES-AND-REPLAY-READINESS.md)：四执行根完整fixture0、KV桶检查失败；actions/latents不等于完整inference重放输入。replay不替代机器人task E2E。此次收口无新GPU性能测试/捕获修复。

全臂512-token BF16优化reset，冷启动另列；horizon800、每局1200秒/4000尝试，无smoke/独立模型测试/SHA。正式性能仍原质量门+本机独占；非测速可按授权共享。仅核依赖清理项目废弃物：此前两失效环境副本2737703749B已删；自然任务失败、在用资产与轻量错误账保留。新CPU资源read辅助临时回传11MB重复日志超自身1MiB帽，超帽真实记录和远端原路径保留，仅本地重复副本已清，最终7288B。

[少步数Target对照](memory/global/requirements/requirement-投机须与少步数Target比较质量成本.md)：T25/T17对T50、video20不变、完整σ网格，先登记，不预设投机获胜。用户最新保持当前配方，四套既定结果齐后再讨论调整；不追加Long tuning或独立TEST。

HW固定v15/legacy，不升级或全量sync。10/22质量/系统go-no-go，10/23候选/API/claim冻结，10/28主表，11/01 18:00中文证据包；论文逐段会话给用户飞书，不创建本地初稿。

[三套实际接续](experiments/libero-four-suite-20261017/BAAI-CONTINUATION-20261018.md)；[首两S解释](experiments/long-morning-20261018/FIRST-TWO-S-AUDIT-EXPLANATION-20261018.md)；[训练/测试与文献](experiments/long-morning-20261018/EVALUATION-PROTOCOL-LITERATURE-20261018.md)。按episode拆FIT/CAL不代表全500独立；同套任务训练/闭环是论文常见做法。新建议待讨论，不改当前协议。
