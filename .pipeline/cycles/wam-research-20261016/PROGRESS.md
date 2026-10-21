# 当前研究进度

更新2026-10-22T02:46:38.498460+08:00。

|ID|阶段|状态|证据|下一步|
|---|---|---|---|---|
|M17|LIBERO证据收口|in_progress|6000局齐，非独立TEST|曝光/首两S/长尾缺口|
|M39|单层先对齐|in_progress|最终完整层约2.296/2.453/2.476ms，有限逐位|NCU及剩余强基线差距，再DAG|
|M40|RoboTwin迁移|in_progress|跨任务4局终态、owned清理|原截止已过，扩大未获新资源|

## M39：夜间已验证版本

[结果](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)、[同轮完整层表](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/WHOLE-LAYER-COMPARISON.csv)、[最终逐阶段表](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/stage-control-final-r2/RESULT.md)。12项候选＋08b修复已终态。选11：静态point/norm分派，padded epilogue，嵌入已有guarded self，B14双4warp attention，FFN1 N256/K32半区写回，删除冗余静态入口同步。仍单层108CTA/256线程，阶段出口依赖保留；无ready-DAG，无30层融合。

B14短/中/长KV同轮两进程CUDA Graph约2.296/2.453/2.476ms，相对本夜开始2.510/2.718/2.977ms下降8.52%/9.75%/16.81%。同工作量uncached guarded强基线2.070/2.187/2.201ms，仍慢约10.94%/12.17%/12.50%；未完全对齐。Torch静态原数学路径2.315/3.101/4.336ms，不挑较慢dense路径宣称超过全部最佳实现。不是机器人E2E或新有效提交成本。

最终9真实签名（B2action/B14action/B2video×3KV）、81完整调用及21结构变体84调用通过有限逐位：原Torch输出、23共有中间量、普通/debug、有效KV更新、输入/RNG。08 short/drop1有2个中间元素差1.49e-8，原FAILED保留；08b非16尾G1修复后通过。不推论所有输入或在线策略质量。源0cc1944b、DSOac3f9e46，[两机部署身份](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/SELECTED.json)。

中KV剩余显著项：self_attention约231 vs guarded157µs，Cross-K RMS116 vs70µs，FFN1 199 vs167µs，FFN2 138 vs117µs，GELU30 vs15µs。阶段独立运行也慢；不是全由全局同步。最小shared/更多CTA控制使RMS/GELU部分改善，但不普遍；完整层100096B shared、255regs/thread、1CTA/SM限制各阶段单独选资源。静态local936B不是动态spill字节。已有grid_constant参数标注，不重复计作新改动。

本机最终版NCU新窗口等待自然空闲，最迟08:47:30启动、09:00清理；前窗口外部干扰0kernel，原PARTIAL保留。 M39累计保守计费4850.877/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。 本轮[资源/停止账本](../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.json)持续显示等待状态，09:00硬截止。远端本轮全部owned已退出。

## M40：工程跨任务收口

[4局结果](../../experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)：basket valid14原版/组合均成功，177.761→166.753秒（−6.19%）；microwave valid14均失败，921.867→887.386秒（−3.74%）。全程受用观察/物理/RNG/命令/RPC摘要与终态一致；固定顺序每例一次探索性成本，不是稳定新质量N。原组合valid16四臂均成功，组合−7.33%保留。全局工程默认未改。

累计54718.243980/86400秒，owned清理；原10/22 01:26:04.706686截止已过，不重启。1800局/160GPU小时未启动/未批准。本机允许最多四分之一、同case同卡、证据回传远端，当前没有本机RoboTwin worker。旧DEV90 T/C/P各30成功25/24/23，非独立TEST；提交调用与实际推进分列，旧字段/罚分不覆盖。

## 共同边界与日期

四项科学权限false；paused72、A_off、旧retention/FAILED、baseline/原tile/投机生产和state/cycle/rules不动。BAAI仅~/cxz；未新建论文初稿。 LIBERO三方法各4×10×50=2000，共6000局；Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499，各500。不是每方法三seed重复。

10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包。质量和系统证据不足就缩小主张，不制造PASS。

[本轮前完整记录](history/overnight-before-20261022/PROGRESS.md)。
