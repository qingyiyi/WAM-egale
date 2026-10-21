---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T22:14:16.314145+08:00'
plan: PLAN.md
current: M40
next: M40
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 入口再精简，9签名及普通/debug有限逐位；性能无稳定退化 | Self-Q编译边界→FFN→DAG |
| `M40` | RoboTwin迁移 | in_progress | 两单项及获准修复组合完整有限等价；组合本轮−7.33% | 跨任务/长失败、重复计时→预算 |

## M39：入口精简与代码带读

[本轮结果](../../experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)、[计时](../../experiments/single-layer-async-gemm-20261020/readability-20261021/WHOLE-LAYER-COMPARISON.csv)。Python直接绑定参数，移除两个无引用设备函数；普通forward不构造完整debug字典/资源统计。CUDA数学、tile、cache语义、sync及错误传播不变，baseline/投机/原tile不动。当前源043c7759、DSO2a8a250e，本机/远端同一验证二进制。

9真实签名（B2action/B14action/B2video×3KV），原Torch输出/23共有中间量/18有效KV转移及输入/RNG有限逐位通过；9组普通/debug返回和状态一致。两个进程三KV整层Graph变化−0.132%～+0.103%，无稳定退化，不称新增设备端加速。完整层约2.506/2.722–2.724/2.974–2.978ms；不是相对Torch新倍数或机器人E2E。

[前投影结果](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)的1.88%–2.08%改善、Self-Q退化与旧快独立参考保留。FFN原语差距未解决，ready-DAG未实现。本轮GPU55.563769秒、CPU构建133.949637秒；M39累计2538.679523/36000秒。owned已退出。[带读prompt](../../../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)按入口→整层→tile→warp→数据流/同步逐段讲。

## M40：两项及组合通过有限工程核验

用户已批准两项优化，并明确允许修复后组合单局补跑；旧组合0动作闭包错误PARTIAL与17.682秒原消耗保留。原作业仍FAILED_OR_PARTIAL，后继独立namespace完成，不覆盖旧终态。[最终结果](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)、[全部完整成本](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/ALL-VARIANT-COSTS.csv)。

同一曝光open_microwave valid16、原Target：原版243.634秒，FRONT_ONLY233.767秒（−4.05%），LAZY_BATCH_ONLY237.787秒（−2.40%），修复组合225.782秒（−7.33%，1.079×）。四臂均完整成功，96观察、368提交、363真实推进、12policy、948teacher。全部受用观察/物理/RNG摘要、命令、server RPC输入输出/RNG/counts及终态相同；并非保存/离线重放所有原始图像或全部KV。

组合setup15.150→10.466秒，get_obs47.081→37.945秒；这两部分合计省约13.82秒，客户端总省17.85秒，RPC102.901→102.830秒基本相同，剩余差异在未单独分解的客户端时段。固定顺序单case一次，不能将7.33%当稳定六任务/投机性能结论。保留为可选工程候选，全局默认未修改；需其他任务、失败长尾与重复对照。原相机/planner固定源码未改，helper只在本进程工作。

本轮原作业781.254秒＋获准补跑255.945秒＝1037.199秒GPU生命周期；M40累计52520.882/86400秒，原10/22 01:26:04.706686截止不变。owned清理，两个GPU均无计算进程。新增160GPU小时未批准，1800局未启动。

旧DEV90 T/C/P各30局成功25/24/23，非独立TEST；所有失败、Target两例及profile保留。旧actual_commands是提交调用，真实推进另列，不覆盖旧罚分。下一步覆盖其他任务/长失败后再估计六任务×100×三臂的吞吐。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED和保护文件不动；BAAI仅~/cxz。LIBERO每方法4×10×50=2000局，三方法6000；不是论文每方法三seed重复。Long T/C/P483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各500。非独立TEST。

10/22阶段go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包；扩大不能完成时如实PARTIAL，不以进度为由放行质量门。[前完整记录](history/readability-and-runtime-before-20261021/PROGRESS.md)。
