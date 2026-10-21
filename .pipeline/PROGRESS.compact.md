---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T15:46:22.848347+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 8warp两候选均退化，已回退4warp | 同前序Cross投影阶段与MMA/shared/local归因，再DAG |
| `M40` | RoboTwin迁移 | in_progress | GPU1固定DEV90运行 | 完整终态与初态/首两S审计 |

## M39：投影8warp已试，未获得收益

[本轮完整结果](../../experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)。用户要求把投影原等待warp用于MMA；先试256×128、8warp，再固定128×128只改为8warp，两个独立进程三KV完整层均退化。大tile慢3.16%–4.85%，固定tile慢1.05%–1.33%；两版各9真实签名/23中间量、有效KV、输入/RNG有限逐位通过max_abs0。仅Cross K/V投影变化，小action投影未试。

大tileshared98848→148000bytes；固定tile仍98848；两版255regs与1CTA/SM未变。两机源码、远端DSO已恢复原4warp。此轮没有新阶段计时/NCU，不能把整层差当投影差或把根因写死。全部负结果/源码/计时/回退证据保留。旧[慢因诊断](../../experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)的NCU计数只对应旧4warp。

本轮GPU79.587378秒、CPU准备266.718863秒，累计M39 **2073.412951/36000秒**。0异常/0超时，2性能负结果；自然退出/回收，GPU0已释放。本轮两job分别16:34:29/16:39:44截止，均提前完成，旧期限不改。单层重放，不是机器人正式E2E；四权限false。

## M40：RoboTwin后台

15:40快照完整33/90，前11组C/P初态与首两S动作/有效KV摘要/RNG路径均通过。microwave valid14三臂全部失败；basket valid15 T/C成功而P失败（720实际命令、原horizon700）。不能只归PF2数值根因。完整结果以[动态STATUS](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)时间为准。

GPU1单模型worker自动接续，108完整512-token BF16 reset bank；24h累计prior25806.971332秒、10/22 01:26:04截止不重置。基础设施/隔离失败停队不自动重试，CURRENT18/ON部署CAL不替代DEV90质量。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持；baseline/原tile/投机生产与state/cycle/rules不动；BAAI仅~/cxz。LIBERO三臂各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，非独立TEST。10/22 go/no-go，10/23冻结，10/28表，11/01 18:00证据包。

[前完整记录](history/projection-warps-before-20261021/PROGRESS.md)。
