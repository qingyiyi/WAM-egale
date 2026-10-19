---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T22:56:41.237773+08:00'
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
    status: completed
  - id: M34
    status: in_progress
  - id: M35
    status: in_progress
  - id: M36
    status: planned
---
# 当前计划

| ID | 工作 | 状态 | 结果或下一步 |
|---|---|---|---|
| `M17` | 四套既定评估 | 计算完成、审计未放行 | 保留四false与原预算 |
| `M31` | 静态tile子图 | 完成 | 原结果保留 |
| `M32` | 阶段内persistent | 完成 | 有限精度通过，性能负结果 |
| `M33` | 单层退化profile | 完成 | NCU失败、网格反事实已收口 |
| `M34` | 整层统一persistent kernel与warp职责分工 | CPU实现中；GPU待空卡 | 设备原语→warp角色与ready DAG→单kernel→状态与性能门 |
| `M35` | 四套LIBERO质量/长度/独占工程计时 | 运行中 | 原6000局CPU核对完成，新360计时局上限 |
| `M36` | RoboTwin迁移调研 | 待收口 | 任务/权重/初态/兼容性；不启动第二环境 |

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

## M33 — 单层退化profile（完成）

[M33实际profile](../../experiments/single-block-persistent-profile-20261019/RESULT.md)完成。B14长KV静态/阶段内persistent均34 kernel，增量207.072us由self/cross attention与FFN1合计207.2425us解释。108 CTA上限低于理论2 CTA/SM容量；隔离216 CTA反事实使B14短中长相对静态退化从4.154%/4.781%/5.459%降到0.726%/0.474%/0.179%，没有新增加速。不锁时钟，跨case绝对时间非单调，不能作正式性能。

九真实block×四模式两次profile（153调用）、有效三B14反事实112调用，输出/有效KV/RNG通过。NCU权限失败，实测occupancy/L2/DRAM UNKNOWN；两个扩展身份/符号实施失败保留，第三attempt隔离后完成。五GPU job生命周期81.856秒、293 block尝试含失败、0full30/0episode；22:53两GPU无进程，全部owned退出，新证据42.91MiB。baseline及依赖88项不变。

用户要求的是**整个单层统一persistent kernel，内部warp组调度全部tile**。当前M32只有attention/FFN1阶段内常驻循环，QKV/Wo/FFN2仍host cuBLASLt，不满足整层目标。[完整DAG与缺口](../../experiments/single-block-persistent-profile-20261019/WHOLE-LAYER-EXECUTOR-GAPS.md)已登记，整层实现仍待完成。下一M34补设备原语、ready/生命周期与统一执行器，不扩30层，不以新wrapper或多kernel包装宣称完成。

## M34 — 单层统一persistent kernel（未实现）

按上述缺口分设备原语、整层ready执行器、完整精度/状态、强baseline性能四步推进。仅optimized；不修改baseline或投机侧。实际GPU作业另登记有界预算，不沿用M33余额。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

完整旧方案见[本轮前归档](history/M32-before-summary-20261019/PLAN.md)，旧FAILED/数据/数值不改写。

## M35 — 新授权远端工程计时

[四套6000完整局](../../experiments/libero-four-suite-performance-20261019/RESULT.md)已纯CPU核对。用户允许后续性能统一远端，替代最终E2E必须本机；旧质量四false及旧受阻45chunk入口不改写。新[独占工程测量登记](../../experiments/libero-four-suite-performance-20261019/ISOLATED-MEASUREMENT-PLAN.md)：四套各10task/init49、三独立进程重复，360完整episode上限；每卡一个模型worker、两卡并行不同套，6h/卡、1452000尝试、3GiB，总时钟按现场REGISTRATION不延期。原6000质量局不重跑；新重复不增加独立质量N。

GPU0 Goal→Object，GPU1 Spatial→Long。前三套复用原冻结launcher/runner；Long是独立资源/子集schema，完整episode数学/ObservedAPI不改，原500预期supervisor若标PARTIAL保留，新schema仅评估登记10case，不伪造Long500补PASS。所有失败/污染/长尾与出生时间清理保留，阶段partial不自动重试。

## M34/M36 — 用户指定后继

最新用户允许现在CPU编码/编译，与速度评估并行；对应卡计时关闭后验证单层一个persistent kernel与warp职责分工；M33“先全组协作、warp专职可后置”的最终交付解释已被新约束替代。再汇总RoboTwin任务、权重、init/seed数量与迁移要求；当前只调研，不下载权重、安装模拟器或启动新环境。
