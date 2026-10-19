---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T16:53:10.217466+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与算子研究

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Object 1075/1500，T491/C291/P293；2026-10-19T08:51:38Z两活动GPU1，GPU0分片已完/后继排队 | [核对](../../experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 原无效监督/PARTIAL保留，获准14补采已完；本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE，旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block；旧66/AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 两份代码整理＋30层/性能 | `superseded_with_open_findings` | 新108检查过/432full30；最大中位回归2.18%，严格2.5%非全过/NVML异常 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与有限设备原语 | `completed` | 30eager/24Graph过；FFN1+GELU六点两进程严格过，B14延迟降2.45–3.23%；完整device FFN2失败 | [全表](../../experiments/operator-profile-device-20261019/RESULT.md) |
| `M27` | 短KV/B14源码与性能口径核查 | `completed` | 两次资源中断PARTIAL/一次等待到期；未改内核；融合B14约3%为更快 | [结果](../../experiments/operator-shortkv-b14-20261019/RESULT.md) |
| `M28` | 完整block原生baseline与optimized | `completed` | 六actual30/24状态/原grad flags逐位；两进程九点齐，非最强B14/Graph或E2E | [结果](../../experiments/block-native-baseline-20261019/RESULT.md) |
| `M29` | 投机去噪文献重合与远端核验 | `completed` | TS-DP/FlowCast/FLASH直接先例；新增调度仍是假设 | [文献与主张门](../../experiments/speculative-denoising-novelty-20261019/RESULT.md) |

| `M30` | tile/persistent方向与协作形状交付 | `in_progress` | CPU-only核验真实shape/参数/DAG，不先预设tile收益 | [登记](../../experiments/transformer-tile-handoff-20261019/REGISTRATION.json) |

## 当前可见步骤

- [x] 完整边界、固定上游源码/许可证和旧baseline归档。
- [x] 完整native顺序链；真实action中间量/原语、Q128与write/驱逐严格门。
- [x] 研究baseline/optimized接入；最终vector版六query实际30层及安装还原过。
- [x] 最终动态完整block两进程九点性能；原模型839个grad flags兼容actual30验证通过。全部旧资源中断保留。
- [x] 论文原文重合审查、远端只读核验、完整结果/源码讲解、轻量HW更新。
- 后续强B14与候选调度另登记，不借本轮有限通过生产放行；main只发布当前可审阅源码/证据，发布收据独立保留。

## 本轮预算与原截止

M28自10/19 14:54:05至22:39停新/22:54清理；GPU生命周期累计≤3600秒（等待/失败/清理计入）、CPU4/5≤2线程、存储≤1GiB；full30≤360、block≤12000。当前原始账本见M28各OWNED-RESULT；不借M27余额。精度可共享，计时仍无外来PID。

M17原10/17 17:11:31起，10/21 16:56:31停新/17:11:31清理；18723000尝试/15GiB。每卡最多2worker、匹配T200同卡限制，后继自动。M25/M26/M27旧失败与预算均保留，不追溯放行。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

投机源码/配置/运行依赖与原tile只读；本轮按用户新授权可改算子baseline，旧源已归档。1875投机源与保护state/rules复核不变，不建缺失cycle.yaml。不杀他人/改服务/时钟；BAAI写入仅~/cxz；AutoDL不再SSH、关机UNKNOWN。仅main公开镜像。

10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。旧计划和详细经过见[本次归档](history/M28-before-final-native-result-20261019/PLAN.md)。
