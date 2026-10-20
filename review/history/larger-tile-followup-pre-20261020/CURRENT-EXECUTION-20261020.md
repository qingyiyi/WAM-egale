# 当前后继执行结果（不改paused72科学身份）

固定本文件所在commit阅读，不用后续main替代。旧RESULTS.json是paused72原件，四项权限均false，不改历史FAILED。

本轮已将CUTLASS3.5.1 SM80设备GEMM主循环嵌入完整单层，保持一个compute kernel；9真实签名输出/23配对中间量、18有效KV状态转移有限逐位通过。新ready未验证、默认关闭。B14短/中/长整层中位7.9375/8.1437/8.6423ms，同轮Torch/native2.6537/2.9433/3.3851ms，仍慢2.54–3.02倍。不是强cached/Graph、verifier或E2E。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)、[旧当前版profile](../.pipeline/experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/PROFILE-RESULT.md)、[代码与结构](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。源文件cutlass_gemm.cuh及生成器为当前接入点；完整CUTLASS依赖/DSO/fixture不公开，版本和来源在文档中。

当前profile识别旧版GEMM占70.70%–76.08%，其中cross K/V50%–54%。不能将其当作新CUTLASS版profile。下一步应测新主循环/epilogue/local-memory成本，再考虑B14专用化与跨阶段ready。原语复用不等于MegaTransformer创新成立。

RoboTwin原A100/535渲染已修复，114专家初态、108完整bank齐；首六FIT身份Target 5成功1完整失败、83policy/6557teacher。失败actual1520与罚分1500分列。完整任务bank等价未知；TF collector/训练/DEV尚未完整实现。[结果](../.pipeline/experiments/robotwin-pilot-20261020/FIRST6-TARGET-RESULT.md)。

所有本轮owned进程清理，两GPU现场空；算子累计1497/1500block，停止本轮GPU，不重置旧预算/截止。首次CPU编译FAILED及旧数值/资源FAILED保留。下一GPU阶段须有明确有界预算，不自动TEST或30层。

本轮之前的[执行说明](history/device-gemm-followup-pre-20261020/CURRENT-EXECUTION-20261020.md)保留。
