# 当前研究进度

更新：2026-10-20T17:26:40+08:00。当前GPU0/GPU1已空，所有本次owned进程闭合；没有训练或机器人后继自动队列。

| ID | 工作 | 状态 | 终态 |
|---|---|---|---|
| M17 | 四套6000局 | in_progress | 计算齐，科学审计未放行 |
| M31–M38 | 原子图/基线/profile/工程比较 | completed | 正负结果保持，不能拼成E2E |
| M39 | 单层原语与设备GEMM | completed | 当前有限精度通过，速度仍NO-GO |
| M40 | RoboTwin有限迁移 | in_progress | 原Target首六局完整，TF适配/训练/DEV待实现 |

## 当前算子结果

[当前profile](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/PROFILE-RESULT.md)：旧已对齐最终版B14 GEMM占整层70.70%–76.08%，cross K/V占50%–54%，FFN1/2约11%–12%。12次完整block、150原语；插桩只作归因，不当正式计时或NCU流量证据。

[固定CUTLASS嵌入](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)：直接在optimized接入SM80设备MmaMultistage，保留整个一层一个kernel、小M路径与BF16分段/bias顺序。9真实签名整层/每格23中间量、18有效KV状态转移逐位通过，输入/RNG/tile通过。新ready没有本次重验，默认False。

B14短/中/长KV新整层中位7.9375/8.1437/8.6423ms，同轮native2.6537/2.9433/3.3851ms；两个独立进程耗时比2.992–3.015/2.683–2.771/2.541–2.558倍。每格4普通样本，六profile trace各1compute kernel。uncached eager完整层，非强cache/Graph、完整verifier/chunk/E2E。前版相对native4–5倍与本轮不属同轮旧/新AB，不能直接相除声称严格增益。

新资源108CTA、255register、82464B shared、1536B localSizeBytes；shared较旧少32.06%，实际occupancy/spill/HBM未知。GEMM的四个compute warp自行搬运/计算，warp1/2空闲；尚未完成专职load/compute全面重叠。原[ready负结果/旧有限通过](../../experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)保留，不转借新路径。

[资源闭合](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/RESOURCE-AND-STOP.md)：累计1497/1500完整block，当前新增8个owned根进程回收，首CPU构建失败保留。原M39 registration240/PLAN900 FAILED不改；本轮停止GPU，余3调用不扩大使用。

## RoboTwin与旧质量

[首六局终态](../../experiments/robotwin-pilot-20261020/FIRST6-TARGET-RESULT.md)：5成功，open_microwave完整算法失败；共83policy、6557teacher returned。失败实际1520命令/罚分1500分列。六局初始物理与观察逐位通过；是FIT身份面板，不是标准成功率或独立TEST。原视频入口0动作PARTIAL和批准补跑保留。

114/114固定专家初态，108条完整512-token BF16 bank；完整任务缓存等价尚未测。原A100/535 rt渲染问题已定位为兼容库误继承并修复，无系统驱动修改。原GPU账本累计4008.739秒；首六局结束，server/clients清理。server局部SERVING是旧快照，不是当前运行。后继TF collector/训练/DEV未完整实现。

[四套LIBERO原结果](../../experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：Long T483/C484/P473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各500。原首两S审计FAILED、训练/DEV曝光和长尾保持；不是独立TEST。旧计时330/360及成功子集结果不用于签发新算子/算法质量。

## 下一步与权限

优先当前CUTLASS版成本分解、B14专用化及高效epilogue，再做ready跨阶段调度。新测量需明确预算；原8h/24h、准备截止与保护继续有效，不自动开30层/TEST/正式性能。原始action保留原算子，baseline/投机/原tile及state/cycle/rules不动。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。原FAILED/retention/paused72/MLP0/22缺收据/default A_off保留。只main公开已核对源码/HW/小证据，不公开模型/PT/DSO/私有SASS/凭据。

10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；[日期核验](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。[本次之前完整进度](history/device-gemm-followup-start-20261020/PROGRESS.md)保留。
