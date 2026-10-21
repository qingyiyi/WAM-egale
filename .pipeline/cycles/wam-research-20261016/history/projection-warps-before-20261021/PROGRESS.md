---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M40
status: in_progress
updated: '2026-10-21T14:40:28.123288+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 三KV阶段/CTA与本机NCU完成 | GEMM映射/local状态及小算子任务粒度；随后DAG |
| `M40` | RoboTwin迁移 | in_progress | DEV90固定队列GPU1运行 | 完整终态与初态/首两S审计 |

## M39：本轮慢因核查完成，GPU释放

[完整诊断](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)、[阶段/等待表](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/TIMELINE-RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESOURCE-AND-STOP-REPORT.json)。当前数学源df2a30724192…未改，0优化候选。完整uncached/eager单层旧两进程结果仍为：短KV慢29.01%/29.32%，中24.68%/24.57%，长14.33%/14.49%。本轮profile不替换正式计时表。

远端三KV420完整调用/10962阶段调用，全部共享中间量、有效KV、输入/RNG有限逐位通过，插桩扰动约1%。native外侧marker与自有调用点边界不同，仅诊断定位。主要差：Cross K/V约1108→1243µs，小投影约234→371µs，FFN约283→367µs，Cross K RMS约69→129µs，GELU约15→42µs。最后CTA到齐后的释放中位约1µs，早到等待含其他CTA的必要计算。

本机真实中KV NCU7范围完成，输出有限逐位一致/有效KV通过；独立hot阶段与完整层分列：

|阶段|native/自有 Tensor活跃|native/自有活跃warp|自有/native DRAM|自有local路径读写|
|---|---|---|---|---|
|Cross K|88.50%/70.29%|12.49%/12.50%|1.691×|27.42MiB|
|FFN1|78.75%/32.90%|12.49%/12.49%|2.083×|27.42MiB|
|FFN2|81.47%/52.85%|6.23%/12.49%|1.194×|24.68MiB|

native这三项local请求均0；自有整层合计328.02MiB local路径请求，不是HBM流量。SASS入口STL/出口LDL支持设备函数保存/恢复成本，不能全部叫allocator spill。自有4warp做MMA，native使用较大输出tile；活跃warp比例相同仍有Tensor效率差，不能仅归occupancy。barrier-stall约47%–51%是warp级指标，不是纯grid.sync墙钟损失。缓存/时钟未控制、NVML空卡利用率残留25%已记录；仅本次诊断，不称正式性能。

下一步先GEMM有效warp/tile及调用live-state，再RMS/GELU静态分派/任务粒度；真实DAG仅后续填尾部。已有noinline函数组织曾改善整层约9%–11%，不能为消local访问盲目撤回。各项独占因果时间、更多fixture硬件计数仍UNKNOWN。

本轮GPU88.361944秒，M39累计1993.825574/36000秒；CPU构建132.990秒，CPU空闲等待886.587秒不占GPU，新增证据约52.7MiB。远端NCU权限拒绝、native Graph捕获、本机DSO依赖、资源admission与一次CPU时间余量检查失败均保留。最终NCU24.244760秒自然退出、root reaped/身份消失、GPU进程为空，原14:40:10截止遵守。没有修改活动数学/保护源。

## M40：RoboTwin后台

14:28快照完整16/90；前5组三臂成功且C/P初态与首两S有效KV摘要/RNG/动作/路径门通过。open_microwave-valid14-P自然失败：1520实际命令、原horizon罚分1500、48chunks；同初态Target正运行，C待完成，不能归因PF2。以[动态STATUS](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)时间为准。

单GPU1模型worker，固定30初态×T/C/P自动接续，108完整512-token BF16 reset bank。CURRENT18的13/18是CONT采集，ON部署CAL不是闭环质量。原导入失败/5.005915秒保留；补启获准，24h累计prior25806.971332秒及10/22 01:26:04截止不重置；后续基础设施/隔离失败停队不自动重试。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持。baseline/原tile/投机生产与state/cycle/rules不动；BAAI仅~/cxz。LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499，非独立TEST。10/22 go/no-go、10/23冻结、10/28表、11/01 18:00证据包。

[前检查点](history/remaining-cost-start-20261021/PROGRESS.md)。
