# 最近Execution checkpoint

## 2026-10-21T14:40:28.123288+08:00 本机NCU完成，慢因证据与资源闭合

- 7范围计数和有限数值/有效KV核查完成；NCU worker24.244760秒、自然退出/身份消失、GPU释放。本轮88.361944秒计入累计1993.825574/36000秒，全部失败保留。
- native/自有Cross K、FFN1活跃warp均约12.5%，自有Tensor管线更低；FFN1 DRAM2.083×、local27.42MiB。整层local路径328.02MiB；SASS有函数保存/恢复，不能全部叫spill或HBM。
- 三KV阶段/CTA数据有限逐位通过，主要差在GEMM、Cross RMS/GELU内部；长等待含其他CTA有效工作。没有新增优化候选，也未实现ready。
- [完整分析](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)、[NCU原件](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/local-ncu-exclusive/NCU.csv)、[资源](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESOURCE-AND-STOP-REPORT.json)。
- RoboTwin16/90出现microwave PF2自然失败，T/C配对未齐，不归因；后台继续、截止不变。


## 2026-10-21T14:20:56.166173+08:00 当前三KV阶段/CTA诊断完成，本机NCU待自然空闲

- 420完整/10962阶段调用，有限逐位/有效KV/输入/RNG通过；插桩扰动约1%。[结果与边界](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/TIMELINE-RESULT.md)。
- 差距集中在Cross K/V、小投影、FFN及Cross RMS/GELU。不能把CTA计算尾部解释为纯grid barrier成本；最后到齐后释放中位约1µs。
- 远端NCU权限失败、native Graph捕获失败、本机DSO导入失败及资源admission阻塞均保留；修复仅测量wrapper。64.117185秒GPU计入累计1969.580814/36000秒。
- 本机当前lhw PID45224在用。用户最新允许自然空闲即profile，CPU watcher已运行，≤480秒采集+清理；原14:40:10截止不重置。
- RoboTwin14:18完整15/90，前五组三臂通过首两S审计；GPU1继续open_microwave-valid14-P，计数推进。非独立TEST、四权限false。
- Maintain已合并自然空闲可直接启动NCU的最新授权，替代之前等确认的临时要求。

[前执行记录](history/remaining-cost-start-20261021/EXECUTION.md)。

## 2026-10-21T15:34:34+08:00 投影8warp实现与有界验证启动

- 现有M39直接修改optimized：Cross K/V TM256时8warp参与加载/MMA，WarpTile64×64、barrier256；其余路径保持4warp/barrier128，旧before源码/DSO保留。
- CPU2线程编译中；随后9真实签名有限门与两进程三KV对照自动接续。GPU≤540秒、存储≤1GiB，硬截止16:34:29；旧累计1993.825574秒不重置。
- 远端DEV90 15:32快照29/90，原GPU1链继续；microwave valid14三臂均失败，basket valid15仅PF2失败。待全量汇总，四权限不变。
