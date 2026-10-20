# 当前短入口

更新：2026-10-20T16:24:50+08:00。先读INDEX/CONTEXT-POLICY/约束记忆，再按当前Cycle读取四文；不展开全旧索引、trace、大JSON。[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)。

## 已观察结果

[M39最终单层](experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：9真实签名×阶段/ready两路径18格，最终输出、可配对中间边界逐位一致；每路径6起点×3个合法KV更新=18次转移直接对原Torch一致。输入/RNG/重复/tile审计过，最终27trace各1compute kernel。有限fixture不是任意输入、完整verifier/chunk/闭环质量保证。

当前默认阶段路径仍慢native2.68–5.02倍；FFN行组ready再慢1.2%–6.0%，默认tile_pipeline=False。108CTA×256thread、255registers、121376B shared、1880B localSizeBytes；实际occupancy/流量/stall未知。原LayerNorm设备体、真实cuBLASLt归约/bias顺序修复数值；大M tile64×128×64、小M64×64×32；FA2在单kernel内部。ready有实际交接但其余阶段仍有grid barrier，不能称整层完全异步或H3成立。

[本轮止损](experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)：O1/O2/O3/O4分别261/300、390/400、432/500、297/300，共1380/1500block；43owned根进程回收，GPU worker生命周期324.748秒，CPU编译/总墙钟另列，CUDA active未知。同名DSO两次初测INVALID、PID/清理竞态及全部数值失败保留。旧M39 registration240/PLAN900资源FAILED、旧M34/M39数值FAIL不重写。

[M40 RoboTwin原渲染恢复](experiments/robotwin-pilot-20261020/DRIVER-INTEROP-REPAIR-RESULT.md)：用户恢复排查且不改驱动。固定原场景同脚本，系统535配套libcuda两次完整专家/四路RGB成功，原595 compat路径在相机观察重现OIDN非法访存。已仅修复客户端LD_LIBRARY_PATH，A100/535、rt/32spp/path8/oidn保持。新增119.345秒，原账本累计614.666秒，全部owned已清理；旧两次失败保留。上述修复对照0模型policy/teacher，不是成功率0%；后继运行看下方现场。当前无需因该故障换4090，全任务稳定性待首6条Target验证。

[当前MegaMoE差距审查](experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：已有常驻结构、warp分工、双缓冲和行组ready；仍缺K片段ready与有界中间槽。B14宽GEMM多预留32KiB仅M32使用的partial_y，专用化收益待测；旧M38百分比不是当前profile。

四套旧6000完整episode：每套10task×50init×T/C/P，各臂500。成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。[完整质量与原门](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)。八组合P50/P90在旧界内但max全部>1.30；失败/800罚分、首两S严格配对FAILED、组件失败、训练/DEV曝光和缺匹配500局MLP保留。不是独立TEST或非劣。

M35新同卡工程计时330/360，固定10task/init49×三臂×3独立进程重复；Object1入口0局失败不自动重跑，最多330，原PARTIAL。[结果](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)/[边界](experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。计时重复不增加质量独立N；Long全尝试1.107×/共同成功1.344×、Object1.216×属原工程观察，不借新算子精度签发算法质量。

## 当前资源、下一步与授权

16:24北京现场GPU1原Target第二局运行；首局已完整成功，8policy/240命令、184.197秒，初始物理/观察逐位通过；GPU0仅11MiB无compute worker。[现场](experiments/robotwin-pilot-20261020/resume-evidence/resource-snapshot.json)。114/114固定专家初态已齐。[首case旧视频入口PARTIAL](experiments/robotwin-pilot-20261020/FIRST-CASE-RETRY.md)在0teacher/0动作退出，用户明确允许补跑；first6-3沿原17:46:42北京deadline接续，旧失败与耗时保留。AutoDL归还不再SSH。

用户最新[接续RoboTwin迁移](memory/global/requirements/decision-修复后接续RoboTwin有限迁移.md)，沿[原两卡计划](cycles/wam-research-20261016/ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)。原GPU0 8h/GPU1 24h、准备10/20 18:20北京及各帽不重置；M39收口，M40初态→原Target首6局有界接续。新专家None抓取异常属官方可解性筛选分支，旧FAILED保留，从下一seed继续；已闭合GPU累计2266.544秒，运行中另计。

原始action保留原算子；B14默认短/中/长耗时为native的5.022/4.515/3.956倍。[填尾波审查](experiments/single-layer-async-gemm-20261020/TAIL-FILL-SCHEDULING-REVIEW.md)：空闲CTA取ready任务而非等整轮，关键路径优先；cross K/V可提前但cached时不能重算造收益。当前仅设计分析，无新算子修改/GPU。RoboTwin[首批入口](experiments/robotwin-pilot-20261020/RESUME-FIRST6-PLAN.md)只自动接续原Target六局；TF verifier/采集训练/DEV未完整实现，174局耗时UNKNOWN。

## 冻结接口与科学门

TF64/各套选定权重、normalizer、tau=.007839680526450805、absΔσ×L2、BF16方向/FP32打分、K2B2/bonus/prefix_then_cost等旧身份保持。PF2=SST完整policy请求，T50更新/51 action调用含末尾；reset/cache不计，不是bonus/物理回滚。所有既有可运行LIBERO对照臂同用512-token BF16 reset bank，冷启动另列；RoboTwin已有108条完整bank张量逐位检查，完整任务等价未实测。actual796与失败罚分800分列。

D3072/H24/d128/FFN14336，actionQ16；M32→224有实际aten::addmm证据，attention实例仍Q16。action KV288/1152/2160，video272/1136/2144，cross512；private KV不共享。RoboTwin Q32/16有效动作通道/pool9792需独立接口，不能直接接本算子。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off保留。原M31/M32有限精度不继承给新数学，原M34/M39负结果保留。[主张矩阵](experiments/single-layer-async-gemm-20261020/FOLLOWUP-CLAIM-EVIDENCE.md)：H1有限结构证据，H2本轮冻结7-query对照缺，H3/H4未成立。

## 保护、发布与日历

仅optimized与新RoboTwin独立代码变化；baseline及依赖、既有投机源/配置、原tile/HW、state/cycle/rules保护，不建缺失cycle.yaml，不升级/旧全量sync。BAAI全部写入限~/cxz，私有CUDA12.4/Torch2.9cu126、driver535不改。只main公开审查源码/HW/小证据，无模型/PT/KV/图像/DSO/工具链/私有SASS/连接配置/凭据。

原M17 10/21 16:56:31停新、17:11:31清理及18723000尝试/15GiB，M35 10/20 05:34停新/05:44清理，均属已结束旧作业，不重置/重启。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；正文用户飞书整理，不建本地稿。CVPR官方注册/正文/补充北京11/11、11/17、11/24各19:59，未单列abstract截止；作者profile UNKNOWN。[日期](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。完整旧入口见Cycle history，按需读取。
