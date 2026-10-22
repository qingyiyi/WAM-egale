---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-22T15:31:11.420822+08:00'
plan: PLAN.md
progress: PROGRESS.md
---
# 最近执行检查点

1. Maintain保存先融合self RMS+RoPE决定，仅本次单因素。
2. 直接修改optimized；保留原归约/BF16/FP64边界及debug中间审计，normal移除SQN/SKN物化。
3. CPU编译308.493秒；活动before源码/SASS确认投影及attention QK/PV使用BF16 Tensor Core。
4. 9签名81次、21结构84次有限逐位/状态通过；两进程各9408完整层调用，短/中/长延迟降0.395%/0.421%/0.439%。
5. 15:28:28 owned全退出，无GPU失败；整窗153.757秒，累计7480.439秒。早期收尾拒绝记录保留。
6. 两机源0f79579d/DSO795be814采用；旧已验证版保留，完整层仍慢最强guarded约9%–11%。
7. 用户要求先核查Cross K/V复用，并追加online RMS调研。源码及三档已有压缩context的CPU审计完成：每档两种CFG条件，三档根相同；组内重复为原编码含义，未冒充原GPU14行重验。新GPU调用为0，源码/DSO未变。当前RMS已online累计平方和，post-GEMM融合/新归约尚未验证。

[结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)、[前检查点](history/before-rms-rope-result-20261022/EXECUTION.md)。
