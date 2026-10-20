# 最近执行 checkpoint

2026-10-20T23:19:23+08:00；cycle wam-research-20261016；[PLAN](PLAN.md)，current M39。

1. 用户提升算子累计预算10h；随后优先简易版、同速则仅维护简易版，已按Maintain保存。
2. 简易版完整9签名精度/状态通过，复杂版独立重建对照；修正invalid slot比较错误后两进程无稳定性能退化。旧失败/计时保留。
3. 30阶段×三KV的A/B/C矩阵完成；原版副本baseline冻结，29有效阶段逐位，B14 cache_insert N/A。
4. 当前仅简易活动入口，源码中文注释、外部核验。没有改变数学精度或正式权限。662/720block、4524stage，18owned根退出，GPU0空。
5. 截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。GPU1自动Target采集不受本轮清理影响，截止/总帽不变。

[结果](../../experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/stage-alignment/RESOURCE-AND-STOP.md)、[旧checkpoint](history/simple-first-start-20261020/EXECUTION.md)。下一步先pointwise/layout，再Norm/FFN1/cross-FA2，后ready。
