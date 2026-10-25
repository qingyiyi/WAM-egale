> 最新：双CTA仍NO_GO保留A；用户新要求本机修复接续，resume05已启动并行准备/传输，单独截止21:56:53；远端两卡RoboTwin继续。

# WAM研究短入口（2026-10-25）

先读[约束](memory/INDEX.md)，再按需读[当前Cycle](cycles/wam-research-20261016/PLAN.md)。本文件是派生摘要；原始PROGRESS、保护文件、旧FAILED/retention、原tile不改。当前两个active Cycle时本Session聚焦WAM，不涉及Paseo语音。

M41已收口：[结果](experiments/rtx5090-spec-speed-20261025/RESULT.md)、[五环境表与账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)、[固定源码/原数据入口](experiments/rtx5090-spec-speed-20261025/SOURCE-AND-RESULT-ENTRY.json)。三臂T/TF/PF2，有限暴露DEV工程面板，非独立TEST/正式性能。

| 环境 | N | 成功T/TF/PF2 | 动作加速TF/PF2 | client任务E2E TF/PF2 |
|---|---:|---|---:|---:|
| long | 8 | 8/8/8 | 1.233/1.146 | 1.121/1.062 |
| goal | 10 | 10/9/10 | 1.161/1.137 | 0.828/1.123 |
| spatial | 10 | 9/9/9 | 1.175/1.117 | 1.065/1.013 |
| object | 10 | 10/10/10 | 1.203/1.131 | 1.075/0.990 |
| robotwin | 6 | 5/4/4 | 1.748/1.415 | 0.933/0.936 |

任务E2E包含所有完整失败、每局初始化和关闭，模型已常驻；暖口径、完整policy/chunk、长度与罚分另列原表，不能混用。Goal task9 C失败/P成功；Spatial/Object PF2成功长尾1.697/1.893倍，Object P90也超1.20；Robo篮子新增失败与C订书机2.833倍长尾。Robo五case第二个S在首次PF2刷新前已不同，隔离FAILED、根因UNKNOWN；有限分数/路由通过不能作全KV/velocity等价证明。

Long因两条已执行PARTIAL只有8共同组，旧不同FSDP T0/未配齐T/P1单列；不以成功筛样。原21teacher计数滞后与23尝试/22返回保留，不重跑。Object bank绝对链接失效为0teacher/0episode准备失败，复制原44MiB后新attempt开始。

所有三臂沿优化full512 BF16 bank/eager/no外层Graph；reset实测约0.01%–0.03%总尝试时间。受控资产预传有CPU/IO重叠，Robo首16 shared attention含inline数值检查，尚无正式隔离重复。四字段engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。

M41原T0 01:15:33到06:01:46清理4.770h；95597 teacher尝试/95596返回，1664文件84.02MiB核验/fsync回传后删5090整个临时根/新权重/env/cache。174个birth身份无活跃自有work，GPU compute contexts空；原operator/env存在、实例未关。Goal传输曾因EOF外预分配短时owned36G违反34GiB，已停自有rsync回收并修复，不改cap/旧日志。

M40 15:42实查：549/600完整组、1647/1800局，115.389/160GPUh，两guard alive；281/406旧缺口不重跑。[动态结果](experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/ROBOTWIN-PROGRESS.json)。

M39历史DSM8/B与R1数值有限通过，B整层慢49%–65%、R1仍慢31%–45%，不采用；5742调用/31.49分钟，源码与证据已回传。[历史结果](experiments/sm120-ffn-next-20261024/DSM-STREAM/REVIEW_RETURN.md)。

M39四CTA新候选也NO_GO：B4只改cluster启动，C4保留global ACT并局部DSM共享；三KV/27中间量/七段/有效KV/RNG有限通过，但整层比A慢41.12%/48.11%/33.87%。保留A；本轮Torch/A=1.253/1.756/2.186。164CTA使168attention任务增加波次，混合FFN多约0.25ms，不能全部归因DSM；独立无插桩FFN/动态流量未知。7122调用、28.98分钟，56文件回传/清理闭合。[结果](experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/REVIEW_RETURN.md)、[17阶段表](experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/STAGE-TABLE.md)。

H02本机轻量化完成：25564文件逐项校验/接收端fsync后归档释放120.619GiB，runner退出；另有公共组件13.2068GiB去重和/tmp清理分别记录。默认TMPDIR=/rjs/cxz/tmp，原tile/代码/轻量失败/代表输入保留。[结果与恢复](maintenance/storage-placement-20261024/RESULT.md)。A100质量/5090性能/本机轻量开发；5090本轮无权重常驻，后续需重传。

后续：M40完整覆盖与原微波炉诊断；Robo分叉前一致性定位；再据质量冻结候选/正式性能，不能盲目加训练。旧M17配对/FIT-DEV缺口仍在，M01–M04历史完成不代表整体完成。完整旧短入口在PROGRESS.compact.before-M41-20261025.md；Cycle四旧文同目录before-M41保留。

M42原15:45:32窗口已收口：resume03在0teacher/0新episode因Goal VAE路径缺config失败；supervisor收尾误终止自身造成RUNNING旧状态。独立核对已归档204结果文件/fsync、清理本轮临时资产12.12GiB；不重跑/不延长。原10完整/7初态无效/1中断，0合格四臂组，不能判优阈值。[终态](experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume03/RECONCILE-RESULT.md)。

M39慢因追加核验：A仅170→152CTA，整层慢4.5%–15.5%；168任务的Cross约+67%–69%、中KV Self+74.6%。DSM新计时READY72–73μs、复制/交接38–39μs，不能称全为同步或纯MMA。A FFN尚慢15%，整层仍快1.25/1.76/2.18倍；保留A，下一步定位其25.8μs主循环包络。[归因与全表](experiments/sm120-ffn-next-20261024/CTA-ATTRIBUTION-20261025/REVIEW_RETURN.md)。

M39新增归因作业13:10闭合：5409调用、约19分钟含编译等待与归档，42文件回传尺寸/fsync；自有GPU进程已退出。原A未改，下一工作是主循环进一步归因，不把本轮诊断闭合当作整条算子研究完成。


M39最新双CTA：B2仅cluster启动几乎不变；C2保留M64/global ACT比A慢12.27%–15.46%，G2绕过peer DSM仍慢8.71%–10.42%。R1关键链前序提交与W2就绪等待更多，MMA包络接近；不能称DSM raw指令根因/HBM减半。10413调用/17phase，105文件回传/fsync，owned GPU空；NO_GO保留A。[结果/17阶段表/源码](experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)。

M42恢复04：用户明确失败分析修复并正确跑起来；新T0 15:56:53、截止21:56:53，旧窗口不改。修复Goal/common链接、self-kill，先路径门再模型加载；固定剩余23case/90臂，CPU2+2/单worker，当前准备传输未称GPU运行。[作业](experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/PLAN.md)。
