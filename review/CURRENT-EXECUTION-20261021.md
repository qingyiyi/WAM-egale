# 11:44 更新：GPU1 已恢复 RoboTwin

CURRENT18 已完成4/18、4成功，第5局运行；沿用reset text bank。自动后继为完整数据审计→ON3000→部署接口CAL，DEV90未启动/排队。原24h累计预算、全部历史失败和四权限false不变。此为11:44快照，实时以服务器STATUS为准。

[RoboTwin接续登记](../.pipeline/experiments/robotwin-pilot-20261020/current18-20261021/RESULT.md) / [Cross KV、FFN同前序诊断](../.pipeline/experiments/single-layer-async-gemm-20261020/gemm-context-control-20261021-r2/RESULT.md)。GPU0两进程诊断完成、已释放；同前序未解释Cross KV完整层差距，动态spill归因仍未知。

以下为10:46小算子阶段结果，清理状态属于该历史时点。

---

# 10/21：RMS、RoPE、调制与残差核对完成

固定本文件所在commit审阅；历史paused72与四项false不变。

用户澄清已写入HW：每个子算子使用合适tile大小与自然任务数；108是当前常驻CTA池，不是各阶段必须凑满的tile数。未来按真实DAG就绪从后续任务补位。当前仍逐阶段grid.sync，没有新ready调度。

12个小算子三KV/两独立进程有限逐位通过。适配独立RMS约与Torch相同，融合RoPE/调制/门控残差更快；直接残差略慢。旧“独立阶段”仍带统一执行器与shared预留，不能等同剥离数学体。

活动代码保留point vec4（RoPE保留原FP64）和candidate RMS双行直接读取。两个集成阶段分别通过9真实签名、23共享中间量、18有效KV状态及输入/RNG/重复有限逐位门。

固定60轮预热、交错12轮、两个独立进程，完整B14短/中/长约3.10/3.37/3.62ms，较同轮本次前改善2.2%–3.0%；相对native仍慢46%–47%/40%/27%。这是uncached/eager单层，不是最强cached/Graph、30层、chunk或E2E。所有初轮频率漂移数据保留，不跨轮拼倍数。

新全29阶段和每CTA屏障时间戳已补齐。中KV Cross K/V末尾等待P50约53µs，但最晚到达至释放约2–3µs；早到等待与别的CTA计算重叠，不能给“纯同步占比”或把全部等待当可兑现收益。Self Q与FFN2各96任务，空闲12个worker是未来补位窗口。动态spill、L2/DRAM、occupancy counter仍UNKNOWN。

[完整报告/全阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md) / [所有计时轮次](../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/FULL-LAYER-ALL.csv) / [CTA等待汇总](../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/CTA-WAIT-SUMMARY.csv) / [官方源码核验](../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/OFFICIAL-IMPLEMENTATIONS.md)。

活动源码：[whole_layer.cu](../算子研究/wam_ops/optimized/csrc/whole_layer.cu) / [简易代码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。原FFN2并行独立体接近Torch但整层无稳定收益的负结果不变，未重新启用。baseline、原tile和LIBERO投机生产不修改。

网页审阅重点：先区分Cross K/V与FFN的数学体效率、统一资源/队列与编译影响，再给最小DAG补位方案。cross K/V只依赖context可提前；RMS必须等完整行，FFN2必须等其reduction输入。不能把“填满108”当优化目标，也不能把屏障尾部测量解释成全部同步开销。

本轮GPU生命周期207.379秒，CPU准备/构建507.174秒，M39累计1611.324/36000秒；全部owned进程退出。RoboTwin本轮未恢复，仍是Target48/PARENT18/CHILD/CONT/固定接口CAL完成，CURRENT18/ON/DEV待新窗口。

公开快照仅含源码和小证据，不含模型权重、原始PT/KV/相机、DSO、完整profile或凭据；不能声称仅凭本仓库重算全部真实张量。旧review/RESULTS.json保持不变。

发布说明：两份诊断脚本仅去除了空白行尾空格，原始执行/预检源码逐字保留在本地；原始与展示SHA和变换写入M39-MINOR-PUBLICATION-20261021.json。活动算子源、结果JSON与失败终态不变。

新增[三KV完整汇总](../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/THREE-KV-SUMMARY.md)及逐进程CSV：仅CPU重算展示，三档原实验均已完成。下一步先统一Cross KV/FFN前序访存与缓存条件，现有差值不等于纯同步。
