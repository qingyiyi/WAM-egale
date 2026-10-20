# WAM 投机与单层 Transformer 当前入口

更新：2026-10-20T21:59:48+08:00。

先读[上下文策略](CONTEXT-POLICY.md)、[记忆索引](memory/INDEX.md)及约束，再聚焦[Cycle进度](cycles/wam-research-20261016/PROGRESS.md)和PLAN/EXECUTION/DISCUSSION。v15不升级或全量sync。

- [当前算子profile与源码解释](experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)：24完整block/150原语；GEMM49–57%、Norm/pointwise及FA2嵌入待优化，NCU硬件counter UNKNOWN。现有数学源码未改。
- [前轮普通单层速度](experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESULT.md)：B14三KV约1.73–1.88倍native，有限逐位通过；不是等速或E2E。仍有阶段grid.sync。
- [RoboTwin固定42接续](experiments/robotwin-pilot-20261020/target-remaining42/PLAN.md)：GPU1自动固定轮转，现场10/20 21:59:00北京：新增5/42完整终态、5成功，当前`open_microwave-valid01`；原首六5成功1失败保留。Target48目标FIT36/CAL12，TF训练未启动，截止次日03:37:02。
- [四套LIBERO原质量](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：6000局计算齐，科学审计未放行，不是独立TEST。

下一步：先原版/自有独立/嵌入整层逐阶段对齐，再改主要差距，最后真实ready调度。四项质量权限false，旧FAILED/retention与原始证据保留；baseline/投机代码/原tile/HW/保护文件不动。仅单层，BAAI写入限~/cxz，AutoDL已归还。只main镜像源码/HW/小证据。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；[短状态](PROGRESS.compact.md)。

最新[逐阶段对齐计划](experiments/single-layer-async-gemm-20261020/context128-profile/STAGE-ALIGNMENT-PLAN.md)尚未启动新GPU；当前profile预算已终态。
