# 当前Self QKV有限对照

- [x] 源码与旧Cross8warp证据核查；旧结论不外推Self QKV。
- [x] 只将Q/K/V的warp tile由32×64改32×32，CTA tile64×128、96任务、K64/三流水、两段K1536及BF16提交保留。
- [x] CPU构建和真实9签名/36结构门；失败即停止计时。
- [x] 原Torch/当前4warp/8warp两进程三KV完整单层对照：混合函数8warp慢1.72%–2.02%；隔离函数后快1.42%–1.90%，采用隔离8warp。
- [x] 必要阶段归因、资源清理、HW汇报。

窗口30分钟，沿原累计10小时，旧00:58截止属于上一已闭合作业不重置。只改optimized，GPU0独占本作业、GPU1机器人不动；不访远端硬件计数器，不运行机器人smoke。四权限仍false。改变warp tile可能改变编译布局/舍入，须门检，不能只把if条件放宽到8warp。

同截止后续：隔离函数r3、两进程phase控制、原Torch所有阶段重测、两机采用和owned清理已完成。旧attempt及FAILED未改。
