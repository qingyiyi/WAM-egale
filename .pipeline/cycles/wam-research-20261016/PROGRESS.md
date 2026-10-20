---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
status: active
updated: '2026-10-21T02:34:36.439687+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 结果/证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO | in_progress | 6000局齐，科学审计未放行 | 保留曝光/首两S/长尾缺口 |
| `M39` | 单层对齐 | in_progress | 本轮有限精度通过；性能未对齐，归因报告齐 | 原语优先，不启动ready |
| `M40` | RoboTwin | in_progress | Target48齐、parent17/18齐，GPU已退出 | 缺一FIT补跑待答复 |

## M39

用户要求先对齐再调度，不能用新调度掩盖原语差距。活动源码为RMS向量搬运后的顺序版；
既有ready试验仅保留历史，当前逐阶段grid.sync。仅optimized简易单层，baseline/原tile不改。

九真实签名（B2 action/B14 action/B2 video×短中长）输出、23共享中间量、18有效KV转移、
输入/RNG/重复有限逐位误差0。恢复源码后阶段复核也通过，不外推任意输入或机器人。

两进程B14完整层：短native2.6307/2.1263ms，自有3.8656/3.3265ms（1.469/1.564倍）；
中native2.4064/2.3986、自有3.5627/3.5607（1.481/1.485倍）；
长native2.8402/2.8391、自有3.8112/3.8151（1.342/1.344倍）。
保留全部轮次及1095→1410MHz状态变化；不是最强cached/Graph、30层、chunk或E2E。

[最新版阶段与原因](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md)：中KV独立阶段cross K/V约1.06、cross attention1.11、
FFN1 1.21、FFN2 1.41、cross-K RMS1.65、table2.58、GELU3.40倍native。
self attention阶段约0.78倍，但原生含历史pack，不能称纯FA2核心加速。
B仍含cooperative队列/同步/98848B shared预留，不是纯算术成本。
C−B还包含不同编译资源/cache历史，不能当纯barrier。
同SM周期辅助记录表明中KV关键B/C约1.40GHz，不能仅以频率解释其差距。

整层256线程、255寄存器、98848B shared，1CTA/SM、108CTA。
RMS直接向量搬运保留（完整层改善1.1%–1.7%）。
N256、point8192、强制内联、compact B14均撤回；compact虽2CTA/SM但整层慢2%–10%，
其中FFN2和小投影退化、attention改善。不是已证明单kernel无法对齐；动态NCU/spill流量未知。
全部源码/证据已本地汇总；远端/本地关键源摘要一致。

本轮5970/6100 block、13572/14000 stage；GPU worker生命周期619.984秒，
CPU编译/准备3980.981秒；CUDA active未知。M39累计约1285.849秒/10小时。
全部登记子进程退出，两卡终态无GPU计算进程。[资源与失败](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESOURCE-AND-STOP.md)。

## M40

Target48终态41成功7自然失败，PARENT2000与一次CAL齐。parent剩余17现13成功4自然失败，
259policy chunks、12852teacher尝试/返回；只是采集，不是DEV质量比较。
原place_object_basket/valid08为0动作接口错误PARTIAL；修复后r3八完整policy有限门通过，
补跑仍待用户答复。固定18 CPU审计仅FIT11/12（5494行）、CAL6/6（3078行），不出完整训练包。
后继CHILD/CONT/ON、current18、三臂DEV90未启动；不改固定分区、不自动重试。

17局按原03:55截止前结束，owned清理为空；累计GPU生命周期5.6191/24小时。
原失败、22缺收据和未知计数不补造。108完整512-token BF16 bank启用，
RoboTwin完整episode bank等价未测。[结果与原件](../../experiments/robotwin-pilot-20261020/OVERNIGHT-RESULT-20261021.md)。

## 共同边界与下一步

LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。
不是独立TEST；旧A_off、paused72、retention及FAILED不变。
engineering_quality_screen、independent_test_complete、noninferiority_established、
formal_performance_permission均false。

下一步先GELU/table与FFN2阶段对齐，未启动新作业；RoboTwin固定FIT缺口待答复。
GitHub只推main核验源码/HW/小证据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包。
此前记录见[历史](history/alignment-rms-20261021/PROGRESS.md)。
