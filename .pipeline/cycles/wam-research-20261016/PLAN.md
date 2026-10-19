---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T19:27:26.713126+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M28
    status: completed
  - id: M29
    status: completed
  - id: M30
    status: completed
  - id: M31
    status: completed
  - id: M32
    status: planned
---
# 当前计划

| ID | 工作 | 状态与边界 |
|---|---|---|
| M17 | 远端四套既定评估 | Object 1340/1500，完成数T500/C420/P420，来源2026-10-19T11:26:39Z；controller身份匹配=True，原同卡队列继续。 |
| M28 | 完整native block | 完成有限30层/动态block门；旧结果不套用于新实现 |
| M29 | 直接先例与创新边界 | 已核验，普通tile/缓存/融合不单列新贡献 |
| M30 | 真实DAG与设计交接 | 完成；按Transformer self/cross/FFN组织 |
| M31 | 隔离optimized与静态tile子图 | [实现与有限验证完成](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)，新full30待资源 |
| M32 | 新full30门与后继调度 | planned，先补资源允许的整模型门/强B14执行边界；persistent尚未实现或启动 |

## 可见步骤

- [x] 固定baseline及实际依赖，88项源/产物/保护文件不变。
- [x] optimized独立Python/C++/DSO，默认sequential；tiled及FFN1开关。
- [x] self/cross/FFN实际子图，attention/FFN1显式device tile入口。
- [x] 9真实block、72状态、输入/RNG/API门、kernel dispatch与独立进程隔离。
- [x] 错误/原attempt/资源归档，81.007秒GPU生命周期，全部owned退出。
- [ ] 新full30：当前约4.9GiB不足约28GiB；尚未取得新证据。
- [ ] 最强B14边界与有界persistent/ready实现，先数值门再性能，不借旧GPU余额。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。


本轮M31用户仅批准前两项实现；[登记](../../experiments/transformer-tile-subgraph-20261019/REGISTRATION.json)CPU2/墙钟3h/GPU30min/512MiB已收口，旧M28/M30账本不重置。上一完整计划见[归档](history/M31-before-summary-20261019/PLAN.md)。
