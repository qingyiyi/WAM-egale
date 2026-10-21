# 最近 Execution checkpoint

## 2026-10-21T18:00:18.327657+08:00 投影写回与整层收益收口

- 原shared写冲突2,064,384→0；padding单因素3.49%–3.66%，四元素搬运再2.75%–2.93%；独立投影距Torch仍4.07%–4.26%。两阶段分别同期对照，不混用两机时延。
- 当前4warp整层短/中/长KV耗时减少3.12%–3.77%；9真实签名输出max_abs0、23共有中间量/有效KV/输入/RNG有限通过。选择源fdf883e及同一远端实测DSO，本地复制；没有新DAG/正式机器人性能。
- vec4首次对照加载相同binary无效，修正paired-r2；首次整层generic dispatch漏模板导致CPU编译失败，r2修复。原件及消耗保留。
- 本轮GPU171.141460秒，CPU编译259.945245秒；累计2305.594158/36000秒。0timeout，owned自然退出/reaped、GPU0释放。
- M40 17:58:29完整57/90，19组审计，GPU1继续；原截止不变。
- [结果/源](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md)、[预算](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESOURCE-AND-STOP-REPORT.json)、[清理](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/OWNED-CLEANUP.json)。

[前完整执行](history/projection-gemm-core-before-20261021/EXECUTION.md)。

## 2026-10-21T19:55:47.071675+08:00 当前源剩余GEMM核对登记

沿M39新增有限测量阶段，十阶段/三KV/两进程，无数学候选改动。此前截止过期不重启；新登记见remaining-gemm-audit-20261021/JOB.json，累计prior2305.594158秒。GPU0入场空闲，GPU1 RoboTwin继续。同步Maintain已确认执行顺序与解释边界。
