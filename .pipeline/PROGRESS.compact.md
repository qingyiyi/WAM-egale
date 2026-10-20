# 当前短入口

更新：2026-10-20T14:18:51+08:00。先读INDEX/CONTEXT-POLICY/约束记忆，再按当前Cycle读取四文；不展开全旧索引、trace、大JSON。[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)。

## 已观察结果

[M39最终单层](experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：9真实签名×阶段/ready两路径18格，最终输出、可配对中间边界逐位一致；每路径6起点×3个合法KV更新=18次转移直接对原Torch一致。输入/RNG/重复/tile审计过，最终27trace各1compute kernel。有限fixture不是任意输入、完整verifier/chunk/闭环质量保证。

当前默认阶段路径仍慢native2.68–5.02倍；FFN行组ready再慢1.2%–6.0%，默认tile_pipeline=False。108CTA×256thread、255registers、121376B shared、1880B localSizeBytes；实际occupancy/流量/stall未知。原LayerNorm设备体、真实cuBLASLt归约/bias顺序修复数值；大M tile64×128×64、小M64×64×32；FA2在单kernel内部。ready有实际交接但其余阶段仍有grid barrier，不能称整层完全异步或H3成立。

[本轮止损](experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)：O1/O2/O3/O4分别261/300、390/400、432/500、297/300，共1380/1500block；43owned根进程回收，GPU worker生命周期324.748秒，CPU编译/总墙钟另列，CUDA active未知。同名DSO两次初测INVALID、PID/清理竞态及全部数值失败保留。旧M39 registration240/PLAN900资源FAILED、旧M34/M39数值FAIL不重写。

[M40 RoboTwin](experiments/robotwin-pilot-20261020/ENVIRONMENT-STOP-RESULT.md)：固定官方源码/Target/资产及私有SAPIEN/CuRobo已准备；原OIDN2.0.1/私有2.3.3均在首个真实场景非法访存，共495.321秒计预算，已清理。0完整policy/teacher、0新采集/训练/CAL/DEV；没有关闭光追/去噪或改相机绕过。根因尚未独立锁定为驱动，需兼容原渲染环境。不是模型失败或成功率0%。

四套旧6000完整episode：每套10task×50init×T/C/P，各臂500。成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。[完整质量与原门](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)。八组合P50/P90在旧界内但max全部>1.30；失败/800罚分、首两S严格配对FAILED、组件失败、训练/DEV曝光和缺匹配500局MLP保留。不是独立TEST或非劣。

M35新同卡工程计时330/360，固定10task/init49×三臂×3独立进程重复；Object1入口0局失败不自动重跑，最多330，原PARTIAL。[结果](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)/[边界](experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。计时重复不增加质量独立N；Long全尝试1.107×/共同成功1.344×、Object1.216×属原工程观察，不借新算子精度签发算法质量。

## 当前资源、下一步与授权

14:12北京现场两卡0MiB/0%、无compute进程，未见owned guard/worker；未来他人状态须动态核对。[现场](experiments/single-layer-async-gemm-20261020/FOLLOWUP-FINAL-RESOURCE.json)。当前无自动提交GPU队列，本机本轮未启动GPU任务。AutoDL归还不再SSH，实例关机未知。

用户已批准[两卡计划](cycles/wam-research-20261016/ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)，替代旧仅规划。GPU0总8h与每stage帽、GPU1总24h/CPU准备10月20日18:20:28北京截止不重置。M39有界后继收口；M40前置阻塞停止，余量不是无界续跑。兼容环境已向用户询问，不购卡/改系统。

下一优先大GEMM/Norm与资源效率，暂不扩大ready队列；完整1/3/7×cache强baseline矩阵仍缺，B6不准入。不能从block重放虚构g或机器人E2E。RoboTwin单局/174局耗时UNKNOWN，先可运行原渲染，再首6局实际估时；新TF接口草稿未训练、tau未校准。

## 冻结接口与科学门

TF64/各套选定权重、normalizer、tau=.007839680526450805、absΔσ×L2、BF16方向/FP32打分、K2B2/bonus/prefix_then_cost等旧身份保持。PF2=SST完整policy请求，T50更新/51 action调用含末尾；reset/cache不计，不是bonus/物理回滚。所有既有可运行LIBERO对照臂同用512-token BF16 reset bank，冷启动另列；RoboTwin尚0policy，不能声称已完成新bank等价验证。actual796与失败罚分800分列。

D3072/H24/d128/FFN14336，actionQ16；M32→224有实际aten::addmm证据，attention实例仍Q16。action KV288/1152/2160，video272/1136/2144，cross512；private KV不共享。RoboTwin Q32/16有效动作通道/pool9792需独立接口，不能直接接本算子。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off保留。原M31/M32有限精度不继承给新数学，原M34/M39负结果保留。[主张矩阵](experiments/single-layer-async-gemm-20261020/FOLLOWUP-CLAIM-EVIDENCE.md)：H1有限结构证据，H2本轮冻结7-query对照缺，H3/H4未成立。

## 保护、发布与日历

仅optimized与新RoboTwin独立代码变化；baseline及依赖、既有投机源/配置、原tile/HW、state/cycle/rules保护，不建缺失cycle.yaml，不升级/旧全量sync。BAAI全部写入限~/cxz，私有CUDA12.4/Torch2.9cu126、driver535不改。只main公开审查源码/HW/小证据，无模型/PT/KV/图像/DSO/工具链/私有SASS/连接配置/凭据。

原M17 10/21 16:56:31停新、17:11:31清理及18723000尝试/15GiB，M35 10/20 05:34停新/05:44清理，均属已结束旧作业，不重置/重启。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；正文用户飞书整理，不建本地稿。CVPR官方注册/正文/补充北京11/11、11/17、11/24各19:59，未单列abstract截止；作者profile UNKNOWN。[日期](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。完整旧入口见Cycle history，按需读取。
