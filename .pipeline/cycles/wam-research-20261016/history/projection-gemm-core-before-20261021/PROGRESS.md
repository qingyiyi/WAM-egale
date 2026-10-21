---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T17:04:44.011067+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 同计算体三调度无显著差别，投影仍慢约11% | 主循环/epilogue及整层资源归因，再DAG |
| `M40` | RoboTwin迁移 | in_progress | GPU1固定DEV90完整45/90 | 自动接续、完整终态与初态/首两S审计 |

## M39：三调度直接对照完成

[本轮完整结果](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md)、[逐进程计时](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/TIMING-SUMMARY.csv)。同256×128/8warp计算体，同compiled kernel比较普通672CTA、108静态循环、108原子队列。两进程、三真实B14 KV、K/V两投影同前序：均约615µs，原版stage约553µs。静态/原子相对普通中位数差≤0.17%；原子含清零总区间增加0.17%–0.33%。没有证据将原版优势归于CTA总数或领任务，独立投影仍慢11.09%–11.30%。

36组真实投影逐位通过max_abs0，672tile各一次；完整层23共有中间量+output、有效KV/输入/RNG有限通过。仅B14单层fixture，不外推机器人或所有输入。当前4warp整层源码df2a307与DSO保持不变；大tile8warp再次完整层慢3.11%–5.00%，不选用。

独立编译254regs/local0，对比插桩整层255regs/local1456。但插桩本身增加1.32%–1.40%，阶段多15–19µs不能全归于local或同步；本轮没有新NCU，不把旧4warp计数当8warp根因。后续先核对GEMM主循环/写回，再就绪DAG，避免继续无依据扫CTA/warp。

本轮GPU61.039747秒（包含两次脚本失败23.611243秒）、CPU编译187.094542秒；M39累计**2134.452699/36000秒**。0timeout，owned自然退出/reaped，GPU0释放；新证据26.73MiB。原18:14:07截止不重置。[失败与修正](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/HARNESS-CORRECTIONS.md)、[资源清理](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESOURCE-AND-STOP-REPORT.json)。旧8warp两候选负结果、全部FAILED保留。

## M40：RoboTwin后台

17:02:19快照完整45/90，当前lift_pot-valid16-T；guard与队列正常，已完成15组C/P初态/首两S审计均通过。物理状态/动作/RNG与有效KV摘要范围见[巡检原件](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-170219.json)。原失败如microwave valid14三臂失败、basket valid15仅P失败保留；不能只归PF2数值根因。

GPU1单worker自动接续，108完整512-token BF16 reset bank。24h累计prior25806.971332秒、10/22 01:26:04.706686截止不变；基础设施/隔离失败停队，无自动重试。数据未齐，不做总体质量放行。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持；baseline/原tile/投机生产与保护state/cycle/rules不动；BAAI仅~/cxz。LIBERO三臂各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，非独立TEST。10/22 go/no-go，10/23冻结，10/28表，11/01 18:00证据包。

[前完整记录](history/projection-dispatch-before-20261021/PROGRESS.md)。
