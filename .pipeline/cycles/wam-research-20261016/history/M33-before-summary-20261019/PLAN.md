---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T21:08:07.097619+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M31
    status: completed
  - id: M32
    status: completed
  - id: M33
    status: in_progress
---
# 当前计划

## M17 — 既定四套LIBERO评估

Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

## M31 — 独立optimized与静态tile子图（完成）

[原结果](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)及旧full30显存不足记录保留。用户随后收缩单层范围，旧full30待补不再作为当前任务。

## M32 — 单层tile/persistent实现与有限验证（完成）

[M32结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)、[两进程九点计时](../../experiments/persistent-tile-remote-20261019/PERFORMANCE.md)。用户最新限定一层完整Transformer block，已取消本轮30层迁移/执行；最终模型E2E后续本机。远端~/cxz隔离CUDA12.4/Torch2.9 cu126编译；baseline与依赖88项不变。

attention/FFN1阶段内实现常驻CTA固定步长tile循环；完整self/cross/FFN与norm/RoPE/调制/残差仍在同一层。QKV/Wo/FFN2仍cuBLASLt，尚无跨阶段ready、warp专职分工或新shared-prefix复用。默认sequential，静态/persistent及FFN1独立开关。本机旧M31 DSO需重编新ABI，远端新DSO已验证。

9真实block（B2/B14 action及B2 Q128 video×三KV）输出逐位；90有效KV连续状态、输入/RNG/参数/API门通过。tile/persistent相对独立顺序路径24中间量，action融合23；48原语检查确认每tile恰好一次与常驻CTA实际多tile执行。无baseline模块/旧DSO加载，原Torch为远端oracle。

两进程378个原始计时样本：persistent+FFN1相对静态+FFN1，B14慢4.42%–6.26%，B2 action慢2.56%–2.80%；未取得新增persistent收益。A/A最大10.07%保留，GPU1/host共享负载及三样本中位限制明确；不是最强B14/Graph或E2E。

五GPU作业77.927秒/731block/48原语/0full30/0episode，全部owned退出。首计时在0block因NVML宿主/容器PID误判退出，修正资源工具后新attempt2/3；传输失败、原收据均保留。CUDA active时间UNKNOWN。原3h墙钟/30min GPU预算收口，不转授下一轮。

- [x] 仅一层完整block；baseline冻结与独立optimized。
- [x] 真实attention/FFN1 tile与常驻CTA循环。
- [x] 九输入/90状态/48原语门；两进程单层探索计时。
- [x] 记录负结果/资源/失败与owned退出；HW及main公开小证据。

## M33 — 单层内部profile与统一执行器缺口（用户已要求执行）

只在这一层中先profile常驻循环的寄存器、驻留和尾部开销，再选一个有公共历史或中间量复用价值的tile分组/相邻子图。静态实现作为当前对照。QKV/Wo/FFN2设备化和ready边界逐项推进，不假设常驻必快，不启动30层大融合。新实验先登记预算，不挪用M32余额。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

完整旧方案见[本轮前归档](history/M32-before-summary-20261019/PLAN.md)，旧FAILED/数据/数值不改写。

最新要求：整层在统一persistent kernel内由warp/协作warp组执行tile，见[约束](../../memory/global/requirements/requirement-单层Transformer统一persistent内核调度全部tile.md)。M32局部实现不等于此目标完成。本轮先执行[M33登记](../../experiments/single-block-persistent-profile-20261019/PLAN.md)，GPU900秒/墙钟60分钟，保持原算术。
