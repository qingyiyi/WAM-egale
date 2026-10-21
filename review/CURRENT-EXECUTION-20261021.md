# 当前执行证据（2026-10-21T23:04:19.543284+08:00）

**RoboTwin两项工程优化已实现，单项和修复组合通过同一完整曝光case的有限等价核验。** 原版243.634秒；相机233.767秒（−4.05%）；未用batch预热延迟237.787秒（−2.40%）；组合225.782秒（本轮−7.33%，1.079×）。[完整结果](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)、[所有完整成本](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/ALL-VARIANT-COSTS.csv)、[工程helper](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/engineering.py)。

- 四臂均96观察、368提交调用、363真实推进、12policy/948teacher；受用观察/物理/RNG摘要、完整命令、全部RPC输入输出/RNG/counts及终态一致。仅同case固定顺序一次，不是所有任务/失败长尾/正式投机E2E；原始全图像/KV未全部持久化，不能声称独立离线重算。
- 相机主要省get_obs约9.1秒，预热省setup约4.7秒；RPC基本相同，其他客户端时段有波动。保留可选工程候选，全局默认未改，下一步跨任务/失败长尾和重复计时，再修订1800局预算。
- 原组合因闭包引用错误0teacher/0动作退出，17.682秒PARTIAL及原job FAILED_OR_PARTIAL保留。修复后用户明确允许新namespace单局补跑，该授权已使用并完成；[原作业](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-20261021/RESULT.md)、[新授权](../.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/PLAN.json)。
- 两作业GPU生命周期合计1037.199秒，原M40累计52520.882/86400秒，原10/22 01:26截止保留，owned清理。**1800局未启动，新增160GPU小时未批准。**

RoboTwin旧DEV90各30局T/TF/PF2成功25/24/23，非独立TEST，旧失败与Target两例复现/profile保留。旧actual_commands为提交调用数，真实环境推进另列，旧罚分不覆盖。LIBERO6000局为三方法四套各500的一轮，非三seed重复。

算子当前仍为源043c7759/DSO2a8a250e，9签名有限逐位/无稳定退化；本轮未新增算子改动，Self-Q→FFN→DAG待做。[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。

四科学权限false，旧paused72/retention/FAILED及review/RESULTS.json不变。前派生commit `bd528959a64fca5543f0197232b9525b1b2887b9`。公开镜像不含大tensor/DSO/rawJSONL/profile；具体遗漏见EXPORT-NOTES-20261021-RUNTIME-OPT.json。
