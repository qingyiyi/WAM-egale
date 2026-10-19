---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T09:39:02.089869+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与系统协同

用户今晨已授权本机GPU空闲即接续完整block验证；Object输出冲突的固定14局补采也已获明确批准。原CPU阶段/失败/预算不改，禁止据旧CPU-only状态重复询问许可。

| ID | 工作 | 当前验收与证据 |
|---|---|---|
| `M16` | Long500组三臂首轮 | 描述性完成，T483/C484/P473；配对FAILED/曝光/四false保留；[完整结果](../../experiments/long-morning-20261018/FINAL-SUMMARY.md) |
| `M17` | Object/Goal/Spatial固定链 | Goal/Spatial各1500局完成；Object14补采/数据准入/ON已完成，匹配T200与全500已接续，每卡一worker；[晨间结果/修复](../../experiments/libero-four-suite-morning-20261019/RESULT.md) |
| `M18` | 迁移原件/环境、释放旧机 | 四套资产落BAAI；AutoDL不再SSH、关机UNKNOWN；[迁移](../../experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md) |
| `M19` | 原算子数据CPU复核 | 有限证据，不外推；[576检查/60门/420轮](../../experiments/operator-parity-evidence-20261018/REPORT.md) |
| `M20` | 本机评测迁新机 | 原Object16有效，原新14完整但监督无效，今晨获准固定14补采；本机不自动接质量；[资源接续](../../experiments/libero-four-suite-morning-20261019/OBJECT-RECOVERY-PLAN.md) |
| `M21` | 原始/B14短中长矩阵 | WITH_STANDALONE_FULL30_FAILURE；旧失败/PARTIAL保留；[旧报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md) |
| `M22` | Torch对齐与有限性能 | 108attention/54block/六full30/三policy严格数值过；旧计时有尾部与资源异常；[结果](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md) |
| `M23` | 新机Codex安装 | 0.162.1已装、登录待用户；不再迁移开发，用户kill模拟器后已退出；[安装结果](../../experiments/remote-codex-setup-20261018/RESULT.md) |
| `M24` | 完整block显式顺序B0 | CPU最终复核/22契约过；今晨六真实fixture×cache的66次GPU逐位/有效状态过；A/A超2.5%停止，性能PENDING；[GPU接续](../../experiments/wam-block-gpu-followup-20261019/RESULT.md) |

## 可见计划

- [x] 昨夜固定ca5f37e源码核验、B0顺序实现、真实完整block/合法写CPU、阶段控制与22项契约；源/结果已发布main。
- [x] 今晨新GPU授权单独登记，66次真实block精度/全部有效KV/输入/RNG和旧GPU gold通过。
- [x] Goal/Spatial各500组三臂回收核对；Object新14capture无效定位、隔离补丁与明确补采授权、新namespace双卡真实接续。
- [ ] 本机低开销等待资源变化，原计时资源≤5%及A/A≤2.5%都满足才做完整配对矩阵；最多一次新清洁窗口尝试。
- [ ] Object14数据准入/ON已完成，监督原500组三臂；每5分钟核验新controller/worker/终态，不自动重试。
- [ ] T25/T17对T50、Long后两轮seed/预算另登记，不自动启动；不先开发persistent scheduler。

## 原预算与当前范围

原M24 CPU≤2/8h、GPU0、08:30停新/08:50清理/09:00交付，其终态不回写。今晨另登记GPU阶段08:48:46起至10:48:46，单A100/CPU≤2/≤5000 block/≤1GiB；118已执行，两worker均自然退出。A/A超门原样FAILED；CPU监控只在真正变为两次资源清洁后最多再启一轮，不为PASS重跑。

M16原10/17 13:55:56起，10/19 13:40:56停新/13:55:56清理；6001000尝试/10GiB，实际1593717，Long已完。M17原10/17 17:11:31起，10/21 16:56:31停新/17:11:31清理，18723000尝试/15GiB。Object批准额外14局/56000次；旧无效14实际5876次已扣，new root remaining cap6106124；新匹配T200用原M17有效总截止，旧T_POOL 10/18截止不重开。v2模板截止启动失败0GPU/0episode保留；v3已双卡运行。运行时以实际登记为准，预算起点不重置。

M21旧1649.274/1800秒及全部失败/PARTIAL保留；M22累计1094.781/1200秒、1002完整model forward、自有退出，不挪余量。完整证据见各Experiment。

所有评测同等512-token BF16优化reset、cold另列；horizon800，actual796和800罚分分列。原tile/保护state/cycle/rules、旧FAILED/NO_GO/retention/22缺收据、4f04 paused72/96与MLP0、默认A_off、四false不变。无新120/独立TEST/新TF正式45chunk/训练扫参/第二环境/撤回融合。BAAI所有写入限~/cxz。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；main只作公开镜像。
