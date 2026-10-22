---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-22T12:18:56.377321+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M17` | LIBERO | in_progress | 6000局齐，非独立TEST | 暴露/首两S/长尾缺口 |
| `M39` | 单层就绪tile | in_progress | 新源码/CPU编译与图审查完成；旧11仍是已验证DSO | 新GPU窗口后验证精度/状态与三KV性能 |
| `M40` | RoboTwin | in_progress | 工程4局终态、旧DEV90 | 截止已过，扩大未获新资源 |

## M39 当前接续（2026-10-22T12:18:20.800737+08:00）

用户最新决定先去掉不必要的全局阶段等待、实现tile调度，替代下文旧“先按阶段编译再DAG”的顺序。已在optimized实现28节点就绪队列：启动仅1次grid.sync，CTA每tile重新领取，Cross-K/V补空闲，FFN2每输出tile有序依赖。普通节点仍按整输入节点就绪，不是完整逐行流水。

[结果](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)：CPU图24配置通过、原数学体抽取比对通过、SM80编译与两机CPU加载通过；新GPU精度/速度均UNKNOWN。首次远端加载缺libcudart搜索路径失败保留，已用既有va环境和runtime路径修正，不安装系统组件。

新DSO3bdf1a62仅留证据，活动DSO仍ac3f9e46。源已同步远端；未启动GPU；原09:00已过，新45分钟窗口待用户确认，累计仍6004.497/36000秒。FFN2减少同tile K分段并行、原子/fence和编译开销可能抵消收益，不能用旧表替新版本背书。M17/M40不启动。

## M39 已观察结果

[晨间结果](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)、[三项对照](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/THREE-TRIALS.csv)、[NCU](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/local-ncu-reference-fixed/RESULT.md)、[LTO阶段表](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/LTO-STAGE-CHANGE.csv)。当前仍选夜间11，源0cc1944b/DSOac3f9e46，两机已核对恢复；没有新候选装入活动路径。

夜间完整层2.296/2.453/2.476ms，相对本夜开始下降8.52%/9.75%/16.81%；同工作量uncached guarded2.070/2.187/2.201ms，仍慢10.94%/12.17%/12.50%。当前晨间重复约2.29/2.46/2.48ms。都是有限输入单层、两进程Graph，不是30层/机器人E2E，也不包括cached projection比较。

晨间FA2 forceinline约−0.05%～+0.11%，路由专化+0.11%～+0.32%，无稳定收益；全局compile+device-link LTO慢15.56%/16.15%/15.67%，拒绝。三个候选各9签名/81调用有限逐位通过，所有失败/负结果保留。原selected11的9签名+21结构变体证据继续有效，不重复声称本轮新跑了21变体。

当前版NCU：local路径243623424B、active warp12.50%、Tensor47.93%、barrier warp stall40.67%；不是HBM同字节，也不是40.67%墙钟grid同步。无外部PID/0MiB准入但NVML残留24–25%，仅诊断。一次旧reference DSO ABI失败保留；已用远端同源码/验证DSO修复本机optimized依赖。

LTO中KV Cross-K/V嵌入约772/768µs，原11为574/570µs；LTO独立对应约565/566µs。FFN1/2反而199→179、138→125µs。插桩完整层扰动0.23%–0.66%。支持编译上下文影响不同阶段；具体动态spill/指令缓存贡献未知。原版本统一资源限制是部分原因，不能全归因全局同步。当前108CTA/256线程、100096B动态shared、255regs、静态local936B；仍没有ready-DAG。

[资源账本](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESOURCE-AND-STOP-REPORT.md)：本段保守计费531.767秒，累计6004.497/36000秒；CPU准备952.151秒另列，CUDA active未测。原09:00不延长；本机/BAAI owned均退出，有限范围止损，不是预算耗尽。两项→三项的新证据前瞻修订保留，不再追加实现。

## M40 与 M17

RoboTwin跨任务4局：basket14原/组合均成功，177.761→166.753秒；microwave14均失败，921.867→887.386秒。受用观察/物理/RNG/命令/RPC摘要同，固定顺序单次探索性成本，不是新质量N。原累计54718.243980/86400秒与10/22 01:26:04.706686截止不变，owned空，不重启；1800局/160GPU小时未启动/未批准。旧DEV90 T/C/P各30局成功25/24/23，非独立TEST。

LIBERO三方法各4×10×50=2000，共6000局：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499（各500）。不是三seed重复，不是独立TEST；曝光/首两S/长尾缺口保留。

## 边界与节点

四项科学权限false；paused72、A_off、retention、旧FAILED、baseline/原tile/投机生产及state/cycle/rules不动。BAAI仅~/cxz，无新论文初稿。10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。下一步见PLAN的M39，不自动启动新GPU作业。

[此前完整进度](history/morning-before-20261022/PROGRESS.md)、[执行中记录](history/morning-inflight-20261022/PROGRESS.md)。
