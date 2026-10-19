---
kind: knowledge
name: 完整native block有限对齐
level: reference
scope: project:WAM-egale;M28-native-block-SM80
status: active
updated: '2026-10-19'
sources:
  - .pipeline/experiments/block-native-baseline-20261019/RESULT.md
  - .pipeline/experiments/block-native-baseline-20261019/RESULT.json
  - .pipeline/experiments/block-native-baseline-20261019/SOURCE-DELIVERED.json
---

M28扩展后baseline/optimized均完整C++/CUDA block，含固定上游Norm/Flash、直接cuBLASLt和自有KV/pointwise；Torch保留Tensor/allocator/stream。不是独立GEMM算法或单kernel融合。原标量pack/slot慢点由profile触发16字节copy+CUB稳定compaction修复。

最终六action各23关键量/6原语、24action/video状态门、12optimized状态门、六actual30 action query在声明有效KV/RNG/参数/安装还原下有限逐位过；原839个可求导参数标志保留。B2 fullquery为真实verifier root CFG0/7派生；video只验证block0，不借旧版本完整policy证书。

两独立进程各9动态block形状、504原始样本齐；相对原动态Torch中位不退化，但A/A最大8.63%，不可把小差值称稳定收益。未比较最强B14 shared-prefix/缓存Graph；当前dynamic native仍有host长度同步，Graph不支持。没有新30层/完整chunk/E2E速度或闭环质量放行。旧FAILED和四false不改。
