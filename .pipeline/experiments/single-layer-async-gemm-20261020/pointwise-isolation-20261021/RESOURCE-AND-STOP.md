# 本轮资源和止损

本轮算子GPU worker生命周期118.096秒，CPU构建368.961秒；等待单列在原STATUS，不冒称CUDA active。
M39累计约1403.945/36000秒，未重置原消耗。本轮828次完整block、7658次独立stage调用，含预热/capture/profile/失败尝试；profile耗时不进入计时表。
各原JOB仍保留独立上限/截止。全部登记子进程已退出，PID+出生身份收据见FINAL-CLEANUP.json；没有杀他人进程。

止损：M128串行FFN2慢；shared中间写回没有稳定大收益；M128并行K虽独立接近Torch，完整层无稳定收益，撤回并保留源/结果。
当前选择table/GELU vec4，完整层尚未对齐。没有开展ready调度、30层、TEST或新机器人性能。

RoboTwin累计5.7624/24小时，详见晨报。保留原PARTIAL、新自然失败、训练语法失败和真实退出码。
