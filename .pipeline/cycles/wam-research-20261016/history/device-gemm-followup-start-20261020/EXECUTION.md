# 最近执行 checkpoint

更新时间：2026-10-20T16:24:50+08:00。

1. 保存用户修复后接续RoboTwin有限迁移的决定，不重置原预算/截止；算子填尾波仅作为待验证建议。
2. GPU1启动固定专家清单。前三任务各19＋lift_pot3条后因官方抓取None断言停止，790.555秒、原FAILED保留。源码核对其属于官方专家可解性筛选的异常分支，接续从下一seed10012开始，不重跑旧seed、不换任务；后续control-7完成114/114并清理。
3. 写独立原Target观察server、冻结初态client及有界guard。完整动作循环AST保持，首case初始物理/观察逐位核对通过并进入真实动作；保留50更新/末尾调用、CFG后真实teacher标签、完整命令与任务终态。按该模型构建完整512 bank，冷成本和等价限制另列。
4. first6-1依赖失败终止，0GPU/模型。first6-2构建108条完整BF16 bank，但首case因可选视频目录仍存在而调用缺失ffmpeg，在0teacher/0动作处PARTIAL；70.743秒和原件保留、owned清理。用户明确允许补跑，独立wrapper移除可选视频目录，first6-3沿原17:46:42北京deadline启动；首局完整成功，8policy/240命令、184.197秒；16:24第二局运行。已闭合GPU累计2266.544秒，运行另计，后继训练/DEV未完整实现。
5. 完成[填尾波分析](../../experiments/single-layer-async-gemm-20261020/TAIL-FILL-SCHEDULING-REVIEW.md)，未改算子或开启新测量。B14 action投影96tile，cross K/V各2688；独立任务可填空，但cached基线必须公平，不能凭此保证追回4–5倍退化。

[实际首批计划](../../experiments/robotwin-pilot-20261020/RESUME-FIRST6-PLAN.md)、[专家异常依据](../../experiments/robotwin-pilot-20261020/EXPERT-SELECTION-CONTINUATION.md)、[现场原件](../../experiments/robotwin-pilot-20261020/resume-evidence/resource-snapshot.json)。

[此前完整checkpoint](history/robotwin-resume-20261020/EXECUTION.md)和[更早执行](history/ordered-followup-20261020/EXECUTION.md)保留，不覆盖旧FAILED或四项false。
