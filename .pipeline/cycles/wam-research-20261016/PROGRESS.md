---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
status: active
updated: '2026-10-21T10:46:42.181704+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 结果/证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO | in_progress | 6000局齐，科学审计未放行 | 保留曝光/首两S/长尾缺口 |
| `M39` | 单层对齐 | in_progress | point/RMS核对与有限集成齐，较本轮前再改善2.2%–3.0%；仍未速度对齐 | 查Cross KV/FFN资源成本，不ready |
| `M40` | RoboTwin | in_progress | Target48、parent18及CHILD/CONT训练/固定接口CAL齐 | 新窗口接CURRENT18→ON→DEV |

## M39

[当前小算子/整层/全阶段及CTA等待证据](../../experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md)。
12小算子真实三KV两进程独立核对逐位通过；适配独立RMS接近Torch，融合RoPE/调制/门控残差更快。
旧独立阶段仍带统一执行器，不代表原语最佳。保留point全vec4（RoPE仍FP64）与candidate RMS双行直接读取；9签名/23中间量/18有效KV/输入RNG有限逐位通过。

固定60轮预热、SM观测1410MHz，两进程当前B14完整层：短3.1032/3.1017ms（native2.1151/2.1197），
中3.3658/3.3737（native2.4011/2.4049），长3.6197/3.6166（native2.8393/2.8381）。
同轮较本次前下降2.2%–3.0%，仍慢约46%–47%/40%/27%，不是强cached/Graph或E2E。
全部未预热轮次保留，不跨轮拼倍数。当前全29阶段表已重测，旧表不再冒充当前。

CTA leader屏障时间戳显示：中KV Cross K/V尾波等待P50约53µs，而最后进入至释放尾部约2–3µs；
Self Q/FFN2各96任务，12个worker可提前等待。等待与其他CTA计算重叠，不能相加成墙钟或称纯barrier。
固定预热后的插桩扰动短/中/长约0.39%/0.37%/1.47%，只属于诊断副本；动态spill/HBM/L2仍未知。

用户明确：每子算子按合适tile大小和逻辑任务数，不凑108；未来按真实DAG就绪填空。当前仍顺序执行，未实现ready。
本轮GPU生命周期207.379秒，累计1611.324/36000秒，CPU准备/构建507.174秒；全部owned退出、两GPU空闲。
baseline、原tile、LIBERO策略与科学门不动。下一步先剩余Cross KV/FFN执行资源差距与任务映射。

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
下一步按小算子结果处理Cross KV/FFN和适配tile；RoboTwin下一窗口接已登记链。GitHub只main小证据，不含权重/原始PT/凭据。
10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00证据包。旧结果见[历史](history/pointwise-isolation-20261021/PROGRESS.md)。
