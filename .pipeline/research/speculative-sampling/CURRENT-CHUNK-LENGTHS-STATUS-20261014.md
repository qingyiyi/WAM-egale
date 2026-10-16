# 当前短中长完整chunk测试状态

> 最新完成（2026-10-14T18:53:35.594+08:00）：[三档完整policy chunk实测](CURRENT-CHUNK-LENGTHS-RESULT-20261014.md)：短/中/长KV加速1.602×/1.544×/1.500×，45完整chunk、18/18正式配对胜出，同算法现场数值/KV/RNG及独立CPU审计通过。两次失败保留，最终累计GPU30550.488秒/36000秒（余90.83分钟），前向收费144550；自有进程清理完成。此前“待办/本次未测得”仅为当时状态，以本结果为准。[MegaTransformer整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)已固定范围，下一步profile和单block原型；tile代码/HW不改，尚无新kernel收益。Knowledge：CHAT-20261014-CURRENT-CHUNK-LENGTHS-COMPLETED。

2026-10-14，Asia/Shanghai。**测试入口与CPU预检完成；实际模型加载时出现外来GPU进程，本轮中止，0个完整chunk、0组性能样本。** 不能以旧长KV结果填充此次短/中/长表，也不能声称三档性能已通过。

用户恢复了本机独占GPU窗口及测试所需CPU权限。本轮创建隔离目录`投机采样/experiments/current_chunk_lengths_v1`，不改原模型、preset、旧实验或tile。原已选择本机资源，不需重问租卡偏好；现在需协调真实连续独占窗口。

## 已准备的测试

同一部署模型，原T对当前preset：A_off、K2/B2、bonus、prefix_then_cost、原WL2阈值、共享KV、O1/O2/O3。不是PF2/MSE，也不是历史BONUS或仅attention核计时。

| 档位 | 真实轨迹frame | 请求前历史KV | action验证总KV | 当前测得速度 |
|---|---:|---:|---:|---|
| 短 | 4 | 144 | 288 | 未测得 |
| 中 | 28 | 1008 | 1152 | 未测得 |
| 长 | 56 | 2016 | 2160 | 未测得 |

短/长使用原保存snapshot；中档按原记录frame4/8/12/16/20/24的video/action/obs顺序重建，计划18次原cache前向，保留原slots和容量，不裁切长KV。都是同task0/init3的真实阶段，范围仅代表性状态，不是多任务统计。

每档RAW_CURRENT数值oracle＋T/current各一次warmup＋6对交替AB/BA正式计时，总45个完整chunk。所有样本保留，中位数、MAD、每对加速比一并报告。相对MAD≤7%作为预声明稳定性检查；采集PASS与稳定性PASS分开，不删除不利样本、不自动补跑。

计时采用CUDA完成同步后的完整`server.infer`墙钟，包含video20/action50、draft、verify、接受/回退/bonus及原CPU输出；reset、状态恢复和验证在计时外，准备成本仍计资源预算。profile/teacher/full trace关闭，无EGL或训练。

数值先验收同算法RAW_CURRENT/current：完整输出、route、50次scheduler提交、实际forward数、RNG、全30层有效KV与metadata。每次恢复检查同一输入和状态；T与近似投机不要求输出相同。显式检查本次O3 `enabled/ok/closed`，避免静默回退被误标为完整preset。

CPU预检验证206份冻结源码/配置、target库存、draft及prefix SHA、fixture元数据、有限累计预算、GPU默认关闭、磁盘余量和交替计划；未导入Torch/初始化CUDA。独立审查未发现启动阻断问题；指出的O3直接证据和非共享臂metadata已补齐，修订前plan/gate仍归档。

## 实际尝试与中止

Job：[p0-current-chunk-lengths-v1-20261014-182138](../../../review/OMISSIONS.md)。1次模型部署上限、5000次forward预约、1800秒硬限（25秒清理预留），累计沿用36000秒和174494次预约收费上限。

连续15次NVML空闲采样通过，18:25:06开始模型加载。18:25:36检测到非本任务PID1801365；检查确认属于用户lhw。父监督器按独占计时约束中止，向已验证所属的本任务进程组发SIGTERM，18:25:39完成退出；没有向外来进程发信号。

加载期间已使用GPU，故不是“0GPU使用”。此时尚未进入完整chunk循环，没有数值oracle或速度结果。保存[FAILED.json](../../../review/OMISSIONS.md)、原server.log、GPU逐次采样、OWNED-PROCESSES和CLEANUP，失败不改写成PASS。

| 资源账目 | 数值 |
|---|---:|
| 本轮GPU阶段计费 | 32.8788586464秒 |
| 累计GPU计费 | 30110.9266973939秒（8.3641小时） |
| 原10小时预算剩余 | 5889.0733026061秒（1.6359小时） |
| 本轮forward保守收费 | 5000 |
| 累计forward收费 | 136964 / 174494 |

模型在计数器闭合之前被中止，按预声明策略计满5000预约；这不是声称实际执行了5000次forward。后续job须继承本次失败账本，不能从上一份成功账本重新起算，也不在本失败job内自动重试。

## 下一步与项目整合

完整chunk测试仍列待办，具体入口已准备。协调约10–20分钟连续独占窗口后，新建有限job、重新封存输入及CPU门，继承本次失败费用后再运行；1800秒是上限而非预计耗时。计时中外来任务出现仍停止，不将共享GPU墙钟写成速度结果。

MegaTransformer的范围和tile复用方案已完成只读审查并形成[整合决策](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)：统一项目目标，保留tile原始代码/HW；首版面向CFG14共享verifier，先诊断一个真实block，再做逐元素融合/工作区复用并扩展30层。当前尚无新MegaTransformer实现或性能结果。

资源窗口恢复后按顺序做：当前完整chunk三档 → 独立verifier瓶颈分析 → 一个block原型 → 完整verify/chunk验收。算法质量协议/已有失败分析可与系统设计并行；正式GPU计时串行。reset继续后置，论文正文只在会话中逐段讨论。
