# 当前研究进度

用户已批准B14 FFN1+GELU、FFN2+残差3写回融合，直接在optimized实现；数学舍入边界保留，baseline/投机代码不改。后续唯一性能对照为原Torch。当前已完成CPU实现，远端GPU0有界作业先编译再有限核验；首编译变量名遮蔽失败保留并修复，r2沿同截止继续。尚未采用新DSO或获得新加速比。

本轮绝对截止2026-10-22 18:12:32北京时间，生命周期最多45分钟，清理预留300秒；计入原累计10小时，历史已用8349.913秒，不重置旧作业截止。[作业](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/JOB.json)。四科学权限仍false。

[PLAN](PLAN.md)；[此前完整结果与未完任务](history/before-ffn-epilogue-fusion-20261022/PROGRESS.md)。此前原Torch同缓存整层比值1.182/1.711/2.765×，当前融合版待测。M17/M40旧限制不变，不恢复过期机器人队列。
