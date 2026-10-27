---
kind: execution
cycle: wam-research-20261016
status: active
updated: '2026-10-27T10:11:52.929047+08:00'
---

# 最近检查点

|北京时间|实际动作与证据|
|---|---|
|10/27 09:42–10:11|CrossKV缓存C0–C4完成：6/6层与8项完整动作/接受路径exact；同缓存Torch/compile公平比较，完整policy已计入构建。3owned退出，runner489.526秒；研究入口保留、生产不切换。[证据](../../experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)|
|10/27 09:46–09:53|缓存四套完成：225输入同进程exact，成本/分布和跨进程路径差异留证；Robo前三依赖失败后首四canonical4/4成功，协议/图像工程审阅通过，四科学权限未变|
|10/27 09:38|Maintain保存flow-based动作模型研究定位；StarVLA-PI/GR00T作为有时间时的后续跨模型验证，未启动；[原文](discussions/20261027-跨模型研究范围.md)|
|10/27 08:58–09:20|用户批准只B7 FFN2连续FP32；三层B1/B7及4真实观测接受/十步传播完成，局部快6%但整层无净收益、72边1翻转，候选不采用。首轮测量INVALID与占卡0启动保留，4owned退出；[本轮](../../experiments/fastwam-ffn2-fp32-20261027/REVIEW_RETURN.md)|
|10/27 08:55–09:13|四套6000局审计完成，T1950/TF1945/PF1944。用户新增耗时分解与共同prompt缓存：分布/阶段图已保存，缓存独立源码/原共同成功输入在5090开始实测。Robo首attempt缺libGL保留60.438秒失败，用户目录补库后同四canonical接续|
|10/27 08:46–08:52|用户要求加速比仅共同成功；Maintain已保存，CPU从原始计时重算四suite，源组N16/19/20/20、trace N3/3/4/4。只替换报告观察器v4，GPU作业不动，旧计时/失败/质量分母保留|
|10/27 07:44|Goal20组DEV三臂均20/20，原10项门通过；新tau=.0145392660、TF接受80.9274%，三卡1500局全量启动，每卡两客户端；5090独占测速接续|
|10/27 07:25–07:35|原controller只读STATUS时SSH255停止，实际110采集局完整。v2接入原状态，不重跑、不重置预算；旧失败留存。5090等待改v3、Robo等待改v3，均未重复实验|
|10/27 07:18–07:20|Object1500局五项审计通过，T499/TF497/PF498；TF新增2败、PF新增1败，2条TF成功长尾保留。全部初始像素配对一致，v2只纠正通用文案错误；本机Robo依赖/资产CPU准备完成，0GPU|
|10/27 06:05–06:23|Object5090 action1.218×/chunk1.064×完成，累计1425.219/7200秒；Robo队列仅调度安全空档并镜像资源扣账，未满足空档所以未启动|
|10/27 05:50–06:00|Object DEV三臂20/20且原门通过后全量；核对算子TMA释放后5090v2有界接续；算子330组eager字节精度通过、未胜compile，未接入投机|

本次短表整理前见[09:32归档](history/before-text-cache-report-20261027-0932/EXECUTION.md)。其余完整动作与失败、早期速度模拟见[本次整理前完整执行](history/before-goal-full-20261027-0750/EXECUTION.md)及其历史链接。当前结果和原始源入口见[PROGRESS](PROGRESS.md)；动态统计见[实时结果](../../experiments/fastwam-full-triarm-20261027/LIVE-RESULT.md)。四科学权限仍false，原预算/截止、失败和冻结源保留。本页不新增GPU授权。
