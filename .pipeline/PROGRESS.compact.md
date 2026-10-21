更新2026-10-21T13:45:15.316595+08:00；当前Cycle详见[PLAN](cycles/wam-research-20261016/PLAN.md)。

# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | FFN1尾块小幅改善，有限逐位通过 | 原语/等待归因后DAG |
| `M40` | RoboTwin迁移 | in_progress | DEV90修复后已在GPU1运行 | 固定完整队列及配对审计 |

## M39本轮完成

[完整表及源码身份](experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESULT.md)、[登记旧模板字段说明](experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/REGISTRATION-NOTES.md)。FFN1前216个M128×N128 tile保留，最后8个拆为16个M64tile。分块及领取组织同时改变；增加内部同步，未跨阶段ready。

9真实签名、每格23中间量、18有效KV转移、输入/RNG/重复输出有限逐位通过，max_abs=0；计时同fixture复核，一个kernel。不是任意输入或完整机器人证书。保持BF16边界，源df2a30724192…已同步本地optimized。

| KV | native ms，两进程 | 当前 ms，两进程 | 改前→本轮改善 | 仍慢 |
|---|---|---|---|---|
| 288 | 2.1146 / 2.1166 | 2.7279 / 2.7372 | 0.504% / 0.243% | 29.01% / 29.32% |
| 1152 | 2.4022 / 2.4069 | 2.9952 / 2.9983 | 0.311% / 0.440% | 24.68% / 24.57% |
| 2160 | 2.8388 / 2.8419 | 3.2454 / 3.2536 | 0.296% / 0.191% | 14.33% / 14.49% |

完整uncached/eager单层，两独立进程固定60预热/12交错；GPU0单worker，本轮计时GPU1无机器人worker；SM1410/MEM1215。profile分开，非强cached/Graph、30层、chunk/E2E。全部样本保留，有10–12/12配对改善，效应较小。

选后同源同前序诊断独立/整层差−1.23%至+0.36%，插桩扰动−0.23%至−0.18%。FFN1调用点约220.16µs；同外侧边界自有223–224µs对native165–167µs，仍慢约34%–35%。原大额集成退化已缓解，不能继续把剩余差全部归全局同步。静态255regs/98848B shared/1CTA/SM/1456B local不变；动态spill、L2/DRAM/occupancy UNKNOWN。仅尾块调整不能解释整个0.4–0.6ms差距。

本轮GPU53.236828秒、CPU准备266.815746秒；累计1905.463629/36000秒。所有owned自然退出、GPU0释放。下一步先当前原语/等待归因，再就绪cross K/V tile补位，各阶段按合适tile，108只是worker池。

## M40已启动固定DEV90

[补启与边界](experiments/robotwin-pilot-20261020/dev90-20261021-r2/RETRY-AUTHORIZATION.md) / [实时快照](experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)。本地快照完整终态3/90，当前`stack_blocks_two-valid14-C`；以STATUS时间为准，不能据此说评估完成。单GPU1模型worker，自动接续固定30初态×T/C/P，不是独立TEST。 首组三臂均成功/240实际命令，C/P首两S隔离门通过；只是1组DEV。

CURRENT18此前13成功/5失败，CONT3000生成FIT12/CAL6；不是ON成功率。5779FIT/3502CAL审计通过，固定ON500updates和11195行部署CAL有限逐位/0接受翻转通过；没有teacher B2/B14或闭环质量证书。

首DEV启动因导入顺序失败，0teacher/0episode，5.005915秒消耗、原FAILED保留。用户明确允许新namespace补启，真实导入链CPU核查后GPU1模型已READY、真实动作执行。沿用108完整512-token BF16 reset bank；原24h累计prior25806.971332秒，截止仍10/22 01:26:04，失败/等待/清理计入。每局需预留1800+900秒；后续基础设施/初态/首两S隔离失败停队，不自动重试。

## 共同边界

LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499；非独立TEST。四权限false、paused72、A_off、旧retention/FAILED保留；baseline/原tile/LIBERO生产与state/cycle/rules不动。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00证据包。

[前完整检查点](cycles/wam-research-20261016/history/ffn1-tail-dev90-20261021/PROGRESS.md)。
