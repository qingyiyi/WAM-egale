# WAM短状态（2026-10-22T12:18:20.800737+08:00）

当前Cycle：[M39](cycles/wam-research-20261016/PROGRESS.md)。先读INDEX/CONTEXT-POLICY/记忆约束，再按需读证据；不全量载入旧索引。

## 算子

用户最新改为优先实现就绪tile。新源码已经完成：B14/Q16单层28节点、1次启动grid.sync、按依赖领取tile、Cross-K/V补空闲、FFN2每输出tile有序发布。CPU图24配置/原数学体比对/SM80编译/两机CPU加载通过；**GPU精度、状态与速度尚未验证**。普通节点仍按完整输入节点就绪，cached路径尚未实现；原子/fence及FFN2并行损失是待测风险。

[新结果](experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)、[源码](../算子研究/wam_ops/optimized/csrc/ready_tiles.cuh)、[实施清单](experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/PLAN.md)。候选3bdf1a62仅证据，默认DSO仍夜间11/ac3f9e46；源码两机已同步，GPU未运行。原09:00已过，新45分钟窗口待用户确认；累计6004.497/36000秒不重置。

旧11有限单层速度短/中/长2.296/2.453/2.476ms，比同工作量uncached guarded慢10.94/12.17/12.50%。9签名+21结构有限逐位通过。晨间3候选均不晋级：两项内联/路由无稳定收益，全局LTO慢约16%。[晨间结果](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)。这些不是新ready结果或机器人E2E。

## 其他主线

LIBERO6000局齐；每方法各4×10×50=2000。T/TF/PF2成功数：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499（每臂500）。非三seed、非独立TEST，曝光/首两S/长尾缺口保留。

RoboTwin旧DEV90成功25/24/23（各30）；工程4局终态、无新增独立质量N。累计54718.244/86400秒，10/22 01:26截止已过；1800局/160小时未启动/未批准。当前无项目GPU作业。

baseline、投机生产、原tile、保护state/cycle/rules不动；旧FAILED、retention、paused72保留。四项科学权限false，BAAI仅~/cxz。10/22 go/no-go，10/23冻结身份，10/28主表，11/01 18:00中文证据包；证据不足收缩claim。
