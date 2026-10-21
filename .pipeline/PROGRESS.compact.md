---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T20:09:29.566093+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 当前源三KV十GEMM核对；bank地址与NCU吻合 | 小投影固定形状设备函数，再FFN原语，最后DAG |
| `M40` | RoboTwin迁移 | in_progress | 固定DEV90完整89/90 | 自动完成末局与终态审计 |

## M39：剩余差距已经区分

[本轮结果](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESULT.md)、[同前序三KV两进程表](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/STEADY-STAGE-COMPARISON.csv)。源码fdf883e及已选DSO未改，新增诊断副本。Cross K/V与FFN独立/整层内差约−1.42%至+0.70%；FFN1自身仍比Torch慢33.74%–35.19%，FFN2慢25.44%–27.43%。六小投影独立慢10.26%–22.22%，整层内相对同设备调用点再增加11.11%–57.50%，是下一优先核对对象。

独立小投影SASS无静态LDL/STL，整层有local访问；固定形状与通用分派生成不同代码，支持专用阶段对照，不能把静态计数当动态spill贡献。原短KV升频样本保留；只补short-steady-r2固定600次预热，主表边界时钟1410/1215MHz。插桩扰动−0.303%至+0.060%。320个阶段输出比较max_abs0，8组case检查有效KV/输入/RNG；有限真实输入，不是新增九签名或机器人质量门。

[bank机制解释](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/BANK-CONFLICT-EXPLANATION.md)：旧8warp独立投影stride128使每128B子事务4行争同8bank；stride136使各行错开8bank。枚举预计写wavefront2752512→688128，额外2064384→0，与旧NCU K/V完全相符。只指所测shared写冲突；不说整层所有冲突归零。当前4warp context-only padding及四元素shared搬运仍是上一轮保留改动；原短中长完整层耗时下降3.12%–3.77%、9签名有限逐位门保留，见[前结果](experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md)。

本轮GPU68.215237秒，CPU准备/编译137.714667秒；累计**2373.809395/36000秒**。0GPU失败/timeout；短KV升频条件问题与远端缺cuobjdump的CPU失败保留，后者改用同DSO本地CPU反汇编。[资源/清理](experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESOURCE-AND-STOP-REPORT.json)。两guard和worker已退出，GPU0释放。新阶段21:24:47截止、旧19:25:53等截止均保留；没有新DAG/数学候选。

## M40：固定DEV90后台

20:07:50快照**89/90**，`open_microwave-valid18-C`，guard活跃，29组C/P初态/首两S审计通过。[巡检](experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-200750.json)。GPU1原队列继续，108完整512-token BF16 reset bank，prior25806.971332/86400秒、10/22 01:26:04.706686硬截止不变，无自动重试；尚未完整归档，不放行。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持；baseline/原tile/投机生产/保护文件不动，BAAI仅~/cxz。LIBERO三臂各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499；非独立TEST。10/22 go/no-go，10/23冻结，10/28表，11/01 18:00证据包。

[前完整记录](cycles/wam-research-20261016/history/remaining-gemm-audit-before-20261021/PROGRESS.md)。
