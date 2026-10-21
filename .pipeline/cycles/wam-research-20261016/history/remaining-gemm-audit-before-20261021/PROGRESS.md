---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T18:00:18.327657+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 投影写回修复，整层缩短3.12%–3.77%，9签名逐位通过 | 剩余FFN/小投影与资源对齐，再DAG |
| `M40` | RoboTwin迁移 | in_progress | GPU1固定DEV90完整57/90 | 自动接续、完整终态及审计 |

## M39：有机制证据的写回改善

[完整结果](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md)、[诊断与来源](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/DIAGNOSTIC-NOTES.md)。本机NCU确认原8warp独立投影每次shared写bank conflict为2,064,384；padding到stride136后K/V均为0，写wavefront减少75%。同前序远端投影615→593–594µs；四元素shared搬运再降至约577µs，距离同期Torch约4.07%–4.26%，尚未完全对齐。SASS为LDS.128、输出仍两条32-bit store，不称为64-bit全局写回。

移入当前4warp完整单层，保持108CTA/阶段同步/数学边界。两进程B14短中长分别约2.660→2.560、2.872→2.774、3.124→3.025ms，耗时减少3.77%/3.42%/3.15%。原Torch直接forward核对9签名：输出max_abs0、23共有中间量、有效KV、输入/RNG通过，并核对B2合法缓存更新。仅有限真实fixture；原始action生产不切换，不外推30层/闭环。

选用源whole_layer `fdf883e`，精确源/二进制身份见[SELECTION](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/SELECTION.json)；本地/远端复制同一DSO，完整门执行在远端。整层regs255/shared98848保持，localSize1456→1464仍更快，不能把该字段当动态spill根因。原8warp完整层退化、三CTA调度差≤0.17%的负结果保留；本轮没有实现ready-DAG。

本轮GPU171.141460秒、CPU编译259.945245秒；累计**2305.594158/36000秒**。vec4首轮误加载同一二进制的对照无效（30.471349 GPU秒），首次整层编译失败（17.848642 CPU秒、0GPU）保留，修正后两进程通过。0timeout，owned已清理，GPU0释放。[资源](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESOURCE-AND-STOP-REPORT.json)、[清理](../../experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/OWNED-CLEANUP.json)。两候选用完，不扩搜索；剩余投影约4%和其他阶段开销仍待解释。

## M40：RoboTwin后台

17:58:29快照完整**57/90**，当前`stack_blocks_two-valid17-C`，guard活跃，19组C/P初态/首两S审计通过。[巡检](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-175829.json)。失败全部保留；不能将成功子集外推总体质量。GPU1自动接续、108完整512-token BF16 reset bank；prior25806.971332/86400秒、10/22 01:26:04.706686硬截止不变，无自动重试。未完整前不放行。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持；baseline/原tile/投机生产/保护文件不动；BAAI仅~/cxz。LIBERO三臂各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，非独立TEST。10/22 go/no-go，10/23冻结，10/28表，11/01 18:00证据包。

[前完整记录](history/projection-gemm-core-before-20261021/PROGRESS.md)。
