# FastWAM全量三臂候选队列

本目录已按夜间授权完成Long工程审阅，01:28启动全量三卡作业。先核验原20组三臂工程条件，失败则不自动扩量。当前Long是首suite，后续Spatial/Object/Goal各自按固定100FIT/10CAL/20DEV配方。

- [x] 固定四suite面板及代码快照、官方horizon和三卡去重分片。
- [x] Long小筛查审阅通过，LONG-EXPANSION-REVIEW.json记录原始摘要SHA与冻结模型/tau。
- [x] Long500组、1500完整episode三卡，完整提交与配对审计通过；T477/TF477/PF473。每卡两隔离EGL客户端共享一模型。
- [x] Spatial、Object分别FIT/CAL训练与DEV达标后完成各500组三臂，完整审计通过。
- [x] Goal同配方通过canonical/采集训练/DEV，500组三臂全量完成：T493/TF490/PF491，完整审计通过。
- [x] [汇总四suite](FOUR-SUITES-RESULT-v1.md)：SR/逐任务/区间/配对胜负/trace/共同成功速度均完成，失败保留；另按用户要求独立开展[共同文本缓存](TEXT-CACHE-v1/PLAN.md)。

全50初态统计包括FIT/CAL/DEV暴露，不称独立TEST。init11–46及49只标本次未用于FIT/CAL/DEV，历史暴露不抹除。TF+PF固定S,S,T，与TF共用冻结checkpoint和tau。所有臂十步scheduler。预算继承OVERNIGHT-EXTENSION-v1并含small已消耗；不动protected状态和算子源码。

2026-10-27 08:46用户更新速度统计要求：只计算都成功的任务。当前三臂主表统一用原DEV测速源组的共同成功交集，完整action/chunk与固定/自然trace均据此CPU重算；源组N16/19/20/20、trace N3/3/4/4。见各suite `SPEED-COMMON-SUCCESS-v1.md/json`和[LIVE-RESULT](LIVE-RESULT.md)。失败不进加速比分母，全部原始计时和全体质量仍保留；观察器v4替换v3，不重启任何GPU任务。

2026-10-27 01:53：continue_suites_v1 PID522780已启动，等待Long完整并审计后接续Spatial/Object/Goal；各suite PHASE-REGISTRATION-v1与部署收据已固定，pilot四个T复用至同卡DEV，新增各110采集＋56DEV局。每次启动从剩余24GPUh/card中扣除全部既有消耗，失败不自动重试；[状态](CONTINUATION-v1-STATUS.json)。

后续suite各自的真实5090速度也已登记同样60观测×5重复＋4组完整policy trace回放，使用各自冻结预测器/tau；累计沿用原2GPUh、已有505.064秒不重置。SUITE-SPEED-v1-STATUS.json记录等待/实际作业。共享协调记录变化会停止自动启动，先核对新算子预约。Long结果不能替代其他suite实测。

完整数据通过审计后，由CPU observer生成各suite `RESULT-REVIEW-v1.md/json`：全体与共同成功trace、逐任务SR、Wilson区间、配对胜负及固定任务内初态bootstrap。属于暴露面板的描述性分析，不改扩量门或科研权限；运行状态 `REVIEW-OBSERVER-v1-STATUS.json`。

2026-10-27 03:36：Long完整结果见[RESULT-REVIEW-v1](libero_10/RESULT-REVIEW-v1.md)及[长尾补充](libero_10/TAIL-REVIEW-v1.md)。TF平均trace接近、共同成功P90比1.021，但4/468长尾>1.3倍；PF5/464、最差2.221倍。Spatial canonical已启动。

2026-10-27T04:27:50.536820+08:00：Spatial100FIT/10CAL完成，固定final3000 checkpoint SHA f800f57d…ed513、新tau=.01513551921；20组三臂19/20、20/20、19/20通过原10项扩量门，1500局全量运行中。独占5090完整action1.19855×/chunk1.05632×，详见[速度审阅](libero_spatial/SPEED-REVIEW-v1.md)；全量质量完成后再下结论。

2026-10-27T05:29:15.717378+08:00：Spatial1500局完整审计，T481/TF481/PF482（每臂500），平均命令110.622/109.960/110.800，TF路径通过79.9533%。canonical较论文Spatial98.2%低2pp，未归因或追溯调整；TF4/475、PF2/474共同成功长度比>1.3，最差1.697/4.645，全部保留。见[全量结果](libero_spatial/RESULT-REVIEW-v1.md)。Object canonical按原登记接续。

2026-10-27T05:59:02.619084+08:00：Object DEV三臂20/20/20，原10项工程条件通过；新tau=.01373200957，500组三臂三卡运行中。5090共享记录变化导致旧队列在GPU前停止，核对TMA05:50释放后[v2有界接续](SUITE-SPEED-RESUME-v2.md)，已耗975.149秒不重置。

2026-10-27 07:18：Object1500局完整审计通过，T499/TF497/PF498；全部均值命令126.958/128.356/127.684。TF接受80.4090%，新增2败，2/497共同成功trace>1.3倍；PF新增1败。完整失败保留，未调整阈值，Goal canonical自动接续。图表v2文字遮挡已保留并仅在v3修复排版；统计值未改。

2026-10-27 07:25–07:35：原控制器在只读远端STATUS时SSH255停止，实际采集继续且110局完整。诊断后[v2接入原阶段](CONTINUATION-RESUME-v2.md)，不重跑episode，原deadline/累计预算/科学代码不变；当前CONTINUATION-v2-STATUS.json，Goal训练完成，DEV接续。5090队列当前SUITE-SPEED-v3-STATUS.json。
