# 当前短状态（2026-10-21T23:02:48.466096+08:00）

先读INDEX/CONTEXT-POLICY/memory约束和当前Cycle四文件；固定v15入口，不运行旧全量sync。原state/cycle/rules、baseline、原tile、投机生产保护不变。四权限false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。paused72、旧retention/FAILED和全局A_off保留。

## M40 RoboTwin

用户同意先做相机/未用batch预热工程优化，并一次性授权修复后的组合补跑；旧组合0动作闭包错误PARTIAL和17.682秒保留。最终原版/相机/预热/组合在同一曝光open_microwave valid16均完整成功，243.634/233.767/237.787/225.782秒；组合本轮−7.33%，1.079×。四臂均96观察、368调用、363实际推进、12policy/948teacher；受用观察/物理/RNG摘要、命令和全部RPC有限一致。未保存全部原始图像/KV，不外推全部输入/任务或TF质量。

相机主要省get_obs，预热主要省setup，两部分约13.82秒；客户端总省17.85秒；RPC基本相同，其余客户端时段仍有波动。固定顺序一次，仅工程信号。保留可选helper，全局默认未改。下一步跨任务/长失败与重复计时，再估1800局。

[结果及代码](experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)、[原作业与失败](experiments/robotwin-pilot-20261020/runtime-opt-20261021/RESULT.md)。本轮两作业GPU1037.199秒，M40累计52520.882/86400；原截止10/22 01:26:04.706686，不因补跑重置。owned清理，两GPU无计算任务。六任务×100×三臂1800目标保留，新增160GPU小时未批准、未启动。

旧DEV90：T/C/P各30局成功25/24/23；30组首两S配对通过，非独立TEST。旧actual_commands是提交调用数；真实推进另列，不覆盖旧罚分。旧90均262.81秒，模型reset12ms/局，18失败占55.11%耗时；原Target两例复现与完整profile留原证据。

## M39 单层算子

当前源043c7759/DSO2a8a250e，两机一致。入口直接绑定参数、移除死函数、普通路径不组装debug字典；九真实签名/普通debug有限逐位，两进程三KV完整Graph变化−0.132%～+0.103%，无稳定退化。前小投影改善1.88%–2.08%及Self-Q退化保留；FFN原语尚慢，ready-DAG未实现。先Self-Q编译边界→FFN→DAG。累计2538.680/36000秒，本轮未新增算子GPU。

## M17 LIBERO与日期

三方法四套各500，共6000局，一轮，非独立TEST、非论文三seed重复。Long T/C/P483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。曝光、首两S严格配对及长尾缺口仍在；不剔除失败换取加速。

10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包。证据不足就保留缺口，不改质量门；BAAI写入仅~/cxz。当前完整入口见[Cycle](cycles/wam-research-20261016/PROGRESS.md)。
