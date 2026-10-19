---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-19T16:09:23.647800+08:00'
---
# 最近执行检查点

## 2026-10-19T16:09:23.647800+08:00：完整native已过有限精度，计时等自然空闲

M28独立C++/CUDA顺序链全部接入baseline/optimized；Torch只读oracle。固定Norm/Flash源码和cuBLASLt descriptor复现，首六action中间量/原语与状态过。Q128缺字段的原attempt失败保留，用3次既有完整canonical policy捕获真实输入并与旧五关键量逐位一致：234个model forward，无机器人episode。

首次full-model装入因FP32调制表/Norm权重主动拒绝，0full30；补支持原dtype后六query/18调用、24block有限门过。独立profile定位scalar pack/slot扫描；16字节复制+CUB compaction后六中间量/六30层/24写读状态重过。所有有效KV/metadata/input/RNG严格，不降低allclose。原模型grad flags只读别名兼容另在新namespace验证。

首scalar9点和最终vector部分计时均原样保留；当前外来PID使双进程矩阵未齐。GPU作业各有PID+birth/上限/自然退出或失败，等待仍计M28生命周期，不动他人。代码/报告落[M28](../../experiments/block-native-baseline-20261019/PLAN.md)。M25 NVML/严格性能、M26 CUTLASS FFN2/NCU偏差、M27资源PARTIAL全部保留。

1875投机源与保护文件复核不变；baseline本轮新授权可改。Object只读巡检2026-10-19T08:06:44Z：T469/C268/P269，2活动，controller identity match=True；原M17预算不重置。

完整此前执行见[归档](history/M28-before-final-native-result-20261019/EXECUTION.md)。
