---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T13:37:18.769422+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与算子研究

| ID | 工作 | 状态 | 证据/科学边界 | next |
|---|---|---|---|---|
| `M16` | Long500组三臂首轮 | `completed` | T483/C484/P473；首两S失败/曝光/四false保留 | [证据](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial既定评估 | `in_progress` | Goal/Spatial全齐；13:31 Object668/1500，T356/C156/P156；3worker活动，后继自动排队 | [巡检](../../experiments/operator-profile-device-20261019/REMOTE-STATUS-READONLY.json) |
| `M18` | 迁移与旧机释放 | `completed` | 四套资产落BAAI；AutoDL不再SSH，关机UNKNOWN | [证据](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | `completed` | 有限576检查/60门/420轮，不外推 | [证据](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评估迁往BAAI | `completed` | 原无效监督/PARTIAL保留，获准14补采已完；本机只接算子 | [证据](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长旧矩阵 | `completed` | WITH_STANDALONE_FULL30_FAILURE，旧失败保留 | [证据](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | 旧独立CUDA attention对齐 | `completed` | 108attention/54block/六full30/三policy有限逐位过；性能异常保留 | [证据](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | `completed` | 0.162.1已装、登录待用户；开发不迁移 | [证据](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 直接Torch复制block | `completed` | 六fixture/18次逐位与状态过；294block；旧66/AA失败不改 | [证据](../../experiments/direct-torch-block-copy-20261019/RESULT.md) |
| `M25` | 两份代码整理＋30层/性能 | `in_progress` | 新108检查过/432full30；最大中位回归2.18%，严格2.5%非全过/NVML异常 | [结果](../../experiments/operator-stable-timing-20261019/RESULT.md) |
| `M26` | 强基线profile与有限设备原语 | `completed` | 30eager/24Graph过；FFN1+GELU六点两进程严格过，B14延迟降2.45–3.23%；完整device FFN2失败 | [全表](../../experiments/operator-profile-device-20261019/RESULT.md) |

## 当前交付与后继

- [x] M26固定合法静态强基线、六真实输入shape/DAG/profile；实际GEMM M32→224，attention每候选仍Q16。
- [x] 一个FFN子图有限验证：完整device FFN失败，保留；收缩FFN1+原BF16 GELU通过六block×两进程，FFN2仍Torch。
- [x] 完整中间量/输出/有效KV/metadata/输入/RNG/参数字节门，全部原值、失败和17个owned作业归档。
- [x] 1093 block host/capture/replay intents，617原语调用，751.141秒生命周期；本机GPU自有进程退出，不自动花余额。
- [x] 本轮报告、源码导航和HW已整理；main发布与最终源码回传在本次会话收尾。
- [ ] 后续先明确完整block可调度边界与FFN2契约方案；P_seq/P_sched未启动，不把约3%通用融合包装成新调度创新。

## 仍有效预算/期限

M17沿原10/17 17:11:31起，10/21 16:56:31停新/17:11:31清理、18723000尝试/15GiB；Object14补采授权不重置，旧无效5876计入。两卡每卡最多2worker，Target200固定同卡配对使部分时段为3活动；已有后继自动排队，不重复重跑充占用。

M26 10/19 12:51:06起、10/20 00:51:06清理，CPU≤2/GPU单worker，帽7200秒/6000 block/20000原语/1GiB；本轮已终态不自动接续。旧M24/M25预算已终止，其FAILED与消耗见各自账本，不挪用。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期。各臂同用512-token BF16优化reset，cold另列；actual796与800罚分分列。

投机源码/配置/运行依赖和baseline不改；本轮1875投机文件复核不变。原tile/shared-prefix依赖原位只读，保护state/cycle/rules不覆盖，不建缺失cycle.yaml。不自动新训练、120/独立TEST、新TF正式45chunk或第二环境。BAAI写入永远仅~/cxz，不杀他人/改服务/手动时钟。AutoDL不再SSH，关机UNKNOWN。GitHub仅main镜像；中文论文会话交飞书，不建初稿。

10/22质量/系统go-no-go，10/23冻结，10/28主表，11/01 18:00中文证据包。旧完整计划见[本次归档](history/M26-before-result-20261019/PLAN.md)，详细结果不重复载入。
