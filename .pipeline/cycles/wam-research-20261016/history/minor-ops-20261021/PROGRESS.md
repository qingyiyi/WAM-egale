---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
status: active
updated: '2026-10-21T09:13:44.589629+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 结果/证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO | in_progress | 6000局齐，科学审计未放行 | 保留曝光/首两S/长尾缺口 |
| `M39` | 单层对齐 | in_progress | vec4整层改善2.7%–3.1%；FFN2独立接近Torch，整层并行撤回 | 先查整层资源条件，不ready |
| `M40` | RoboTwin | in_progress | Target48、parent18及CHILD/CONT训练/固定接口CAL齐 | 新窗口接CURRENT18→ON→DEV |

## M39

[整体与分阶段绝对耗时、源码及同步归因边界](../../experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/TIME-BREAKDOWN-20261021.md)已补齐，纯等待比例仍未知；本次仅CPU汇总。

[最新完整证据](../../experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md)。
真实Torch GELU为vec4 kernel；native代理是标量索引并行kernel。剥离队列和统一资源后，
自有vec4 GELU约比Torch慢4%–6%，调制表更快但包含cast+add融合收益。
固定108 CTA的无队列标量版仍慢，不能归因全是同步。

FFN2去中间全局写回收益小且不稳定；M128串行更慢，M128两K分段并行的独立版本比Torch慢约4%–6%，
全部真实三KV/两进程有限逐位通过。但嵌入整层无稳定收益、短KV慢2%–4%，已撤回。
保留该负结果；不能声称FFN2整层对齐或动态spill已证实。当前仅保留table/GELU vec4，原FFN2串行与逐阶段grid.sync。

当前B14完整层短/中/长约native的1.422/1.439/1.301–1.304倍；同轮较本次前改善2.7%–3.1%。
边界为uncached/eager单层；不是最强cached/Graph、30层、chunk或E2E。全部轮次和频率变化保留。
两个集成版本各通过9真实签名、23共享中间量、18有效KV状态、输入/RNG/重复有限逐位；最终已恢复较快vec4源与二进制。
108CTA、256线程、255reg、98848B shared、1CTA/SM；新FFN2整层local从1464增至1488B，仅静态相关，动态counter未知。

本轮GPU生命周期118.096秒、CPU构建368.961秒，M39累计约1403.945/36000秒。
全部owned进程退出，当前两GPU无计算进程。baseline、原tile、LIBERO投机不修改。

## M40

[晨间结果](../../experiments/robotwin-pilot-20261020/MORNING-RESULT-20261021.md)。
valid08用户批准新60分钟窗口，完整运行自然失败并清理；原0动作PARTIAL不改。
PARENT18现13成功5自然失败，FIT12/6269行、CAL6/3078行审计齐；不是配对DEV。
CHILD2500/CONT3000各500 updates完成，首次训练解析语法失败0更新留档，修复r2完成。
两个权重各10771已有CAL query的GPU adapter/decode BF16逐位通过；tau/normalizer不变，不等于teacher路径或闭环质量。
CURRENT18/ON3000/DEV90未启动；原09:30截止不足整局+清理预留，新单局窗口不延长其他工作。
累计5.7624/24 GPU生命周期小时。108完整text bank沿用；RoboTwin完整episode等价仍未另测。

## 共同边界

LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。
不是独立TEST；paused72、旧默认A_off、retention/FAILED及四质量/性能权限false不变。
下一步先FFN2整层条件归因；RoboTwin下一窗口接已登记链。GitHub只main小证据，不含权重/原始PT/凭据。
10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00证据包。旧结果见[历史](history/pointwise-isolation-20261021/PROGRESS.md)。
