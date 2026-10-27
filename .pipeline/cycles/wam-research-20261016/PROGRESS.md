---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-27T10:11:52.929047+08:00'
---

# WAM当前事实

四套LIBERO共6000局已完成并通过完整数据审计：target1950/2000（97.50%）、TF1945/2000（97.25%）、TF+PF1944/2000（97.20%），缺失/PARTIAL/无效0。见[完整汇总](../../experiments/fastwam-full-triarm-20261027/FOUR-SUITES-RESULT-v1.md)；全量包含FIT/CAL/DEV暴露，不是独立TEST。非测速每卡一模型服务加两隔离客户端；所有失败/长尾保留。07:25控制器只读SSH故障独立接续记录保留，实际采集未重跑。

|已完成全量|target / TF / TF+PF（每臂500）|全体平均命令|tau / TF路径逐步接受|
|---|---|---|---|
|Long|477 / 477 / 473|265.976 / 265.660 / 269.820|.01541235 / 80.4432%|
|Spatial|481 / 481 / 482|110.622 / 109.960 / 110.800|.01513552 / 79.9533%|
|Object|499 / 497 / 498|126.958 / 128.356 / 127.684|.01373201 / 80.4090%|
|Goal|493 / 490 / 491|108.772 / 109.722 / 109.534|.01453927 / 79.9054%|

Long共同成功TF4/468、PF5/464超过1.3倍trace，最差1.749/2.221；Spatial TF4/475、PF2/474，最差1.697/4.645。Spatial canonical较论文98.2%低2pp，根因未定，不能称复现论文SR。详见各suite RESULT-REVIEW-v1和TAIL-REVIEW-v1。全50初态包含FIT/CAL/选择暴露，不称独立TEST。

Object固定final3000 checkpoint `8bbb8e95…d1d4b`；全量TF新增2败/无救回、PF新增1败/无救回。TF共同成功497对中2对trace>1.3倍（最差1.460，初始像素均一致），PF498共同成功无>1.3倍；target比论文100%低.2pp。五项数据审计均通过，缺失/PARTIAL/无效0。后续同配方，未按结果调阈值或改门；[审阅](../../experiments/fastwam-full-triarm-20261027/libero_object/RESULT-REVIEW-v2.md)。

5090速度已按用户08:46补充改为三臂共同成功条件统计：Long/Spatial/Object/Goal源组N=16/19/20/20（各原20），完整action分别1.24109×/1.20897×/1.21848×/1.20718×，完整chunk1.07165×/1.05902×/1.06363×/1.06289×。固定长度TF/PF分别1.06433/1.04348、1.06519/1.03271、1.06183/1.03985、1.06403/1.04212倍，trace配对N=3/3/4/4。使用原DEV测速观测源成功身份；不借全量重复的成功替换。四套各自`SPEED-COMMON-SUCCESS-v1.md/json`为当前主口径；旧含失败结果保留追溯，成功率和全体trace分母不改。CPU重算无新GPU，四suite原测速累计1870.287/7200秒与21:32截止不变。360条B7/B1边Spatial1翻转/0路径变化、其他0翻转，数值检查仍保留全部原输入，不因速度筛选删减。

共同prompt缓存四套独立实测已完成：[最新结果](../../experiments/fastwam-full-triarm-20261027/CURRENT-RESULT-v5.md)。Long/Spatial/Object/Goal action1.2423/1.2046/1.2290/1.1495×，chunk1.0801/1.0677/1.0713/1.0566×；225输入同进程cached/uncached条件、动作、路径与提交计数exact。1125对计时、每臂75输入同步profile，TF文本约.058ms/0.04%、GPU VAE38.7%、动作35.1%。跨独立作业路径仍有4/15/9/23个变化，未定因；Goal慢样本全部保留，不把跨作业变化全归因于缓存。原版6000局不改名为缓存质量。新增1855.370秒，5090原累计3725.658/7200秒，owned退出并释放。

Long重复canonical存在同观测/seed输出不一致和一例原败→成功，根因未定，不能解释掉投机新增失败；[证据](../../experiments/fastwam-full-triarm-20261027/libero_10/CANONICAL-REPEAT-v2.md)。四suite在线末段step7均无两层接受，尚未改算法。

RoboTwin14维权重/stats三处齐备。首四canonical现4/4完整成功，32预测/24执行、十步shift5、eager、strict加载和实际命令前缀及图像工程审阅通过；[结果](../../experiments/fastwam-robotwin-canonical-20261027/CANONICAL-REVIEW-v1.md)。前三attempt为GL/EGL/ffmpeg路径故障、0policy，全部保留；成功232.508秒，含失败与诊断累计502.120秒。所有owned退出。下一步登记Robo自身FIT/CAL/DEV并验证真实14D collector/B7；尚无预测器、阈值、三臂全量或5090速度。4局不代替官方30000局三臂面板。

固定FastWAM代码7faa711、HF8eaceeb、Base+对应stats、action10/shift5。官方实体`wam_tutorial_a100/upstream/fastwam`，投机源码`投机采样/fastwam-base/`，HW兼容链接；视觉prefill一次、固定当前KV，不迁入旧full512/CFG/末端t0。LIBERO及Robo三台资产收据在small-closedloop实验，长期保留。

算子最新：[chunk内CrossKV缓存](../../experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)。算子chunk内CrossKV缓存已完成：6组层、4观测×T/S完整30层/10步动作/路径及条件RNG逐位一致；B1/B7分别建cache，不跨chunk复用。warm层相对同缓存eager为1.407/.925×，相对同缓存compile为.605/.459×；完整policy缓存使compile Target/投机约1.061/1.015×，自有仍慢同缓存compile约19–20%。compile新图动作relL2约.25–.37%，不能继承eager证据。显式研究入口保留、生产未改；5090已释放。

前轮算子完成[B7 FFN2有限FP32对照](../../experiments/fastwam-ffn2-fp32-20261027/REVIEW_RETURN.md)：局部快6%、整层无净收益，4观测72边1翻转/1路径变化，不采用。严格TMA＋1T4M3E仍默认；原INVALID、owned退出与173.362秒成本见原报告和[归档](history/before-text-cache-report-20261027-0932/PROGRESS.md)。未接入投机，不相乘倍数。

失败与历史保留：local初次加载SIGKILL/0episode/349.313秒，低内存T5 meta+assign/mmap后成功；审计初版维度错误FAILED保留；旧Long接受强制模拟728对action1.342×/chunk1.087×不是本轮真实结果；LingBot十步负结果及旧五环境质量均保留，旧队列不恢复。初始4canonical/100FIT/10CAL/20DEV的逐阶段身份、全部失败与旧检查点见[本次整理前全文](history/before-object-full-20261027-0600/PROGRESS.md)和[更早完整进度](history/before-fastwam-overnight-20261027-010743/PROGRESS.md)。

四项`engineering_quality_screen`、`independent_test_complete`、`noninferiority_established`、`formal_performance_permission`均false，全局A_off不变。工程扩量检查不自动放行论文质量。原36h/72 aggregate A100h含既有消耗、24h/card；每job仍有截止、owned清理和保留要求。不写正文、不自动公开发布。
