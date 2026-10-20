# 当前算子开销分析与RoboTwin接续

固定本文件所在commit阅读；旧RESULTS.json/paused72、FAILED、retention和四质量权限false原样保留。

当前context128源码和研究二进制未改。完成24完整block、150独立Torch Linear的有界profile，有限输出/中间量/有效KV逐位通过。GEMM占约49–57%，cross K/V占其中1.88–2.02ms；cross-K RMS0.606–0.654ms，相同fixture native单kernel0.068–0.072ms；长KV self+cross attention约23%。阶段包含队列/barrier，native仅kernel持续时间，不是正式纯算术吞吐对比。

普通速度仍为[前轮](../.pipeline/experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESULT.md)B14三KV对同轮native的1.866–1.868 / 1.836–1.875 / 1.732–1.737倍。uncached/eager、单层，非强cached/Graph或E2E。当前108CTA/8warp，context128×128/K64、action64×128；四compute warp执行CUTLASS/FA2，GEMM epilogue全8warp协作。阶段grid.sync仍在，不是完整ready DAG。

[完整分析与源码行号](../.pipeline/experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)、[资源/owned审计](../.pipeline/experiments/single-layer-async-gemm-20261020/context128-profile/FINAL-AUDIT.json)。NCU查询ERR_NVGPUCTRPERM；动态occupancy/spill/HBM/L2 UNKNOWN，未绕过权限。最新用户反馈：先同输入/同边界对齐原版、自有阶段独立、自有阶段嵌入整层，再优化主要差距，最后context K/V真实ready填尾波；不能把普通缓存包装成新调度贡献。

RoboTwin按用户最新继续指令已在GPU1运行固定剩余42条Target采集。现场2026-10-20T21:59:00+08:00：新5/42完整、5成功，当前`open_microwave-valid01`；原首六5成功1失败保留。这是Target FIT/CAL采集，不是TF评测。目标合并Target48(FIT36/CAL12)，reset复用108完整BF16 bank，完整任务cached/uncached等价未实测。21:37:02启动、次日03:37:02硬截止，6小时含清理；固定队列自动接续，本guard不自动训练。原24GPU小时帽/失败/截止记录保留。[计划](../.pipeline/experiments/robotwin-pilot-20261020/target-remaining42/PLAN.md)、[现场](../.pipeline/experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)。

算子GPU0本轮任务已清理，约23.2秒GPU生命周期；baseline/投机源码/原tile/保护HW未改。此前[快照](history/context128-profile-pre-20261020/CURRENT-EXECUTION-20261020.md)保留。
