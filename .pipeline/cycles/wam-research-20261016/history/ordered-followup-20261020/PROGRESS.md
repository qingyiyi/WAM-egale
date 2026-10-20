---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-20T12:09:06+08:00'
current: M39
next: null
---
# 当前研究进度

2026-10-20 12:03北京资源核查：两张远端A100均空、无compute进程。本轮完成[单卡RoboTwin与算子并行计划](ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)，未安装/下载/启动。新建议与原执行终态分开；预算/里程碑见该唯一计划，不复制为新实验结果。

更新：2026-10-20T11:30:55.620554+08:00。当前范围只有一层；原四套计算和新计时重复分开。[短入口](../../PROGRESS.compact.md)。

| ID | 工作 | 状态 | 终态说明 | 证据 |
|---|---|---|---|---|
| `M17` | M17 | in_progress | 科学门另列 | 见下文 |
| `M31` | M31 | completed | 科学门另列 | 见下文 |
| `M32` | M32 | completed | 科学门另列 | 见下文 |
| `M33` | M33 | completed | 科学门另列 | 见下文 |
| `M34` | M34 | completed | 科学门另列 | 见下文 |
| `M35` | M35 | completed | 科学门另列 | 见下文 |
| `M36` | M36 | completed | 科学门另列 | 见下文 |
| `M37` | M37 | completed | 科学门另列 | 见下文 |
| `M38` | M38 | completed | 科学门另列 | 见下文 |
| `M39` | M39 | in_progress | O1/O2有限逐位通过；O3行组流水验证中 | 见下文 |
| `M40` | RoboTwin有限迁移 | in_progress | 环境/资产就绪；原渲染链阻塞并清理 | [执行](../../experiments/robotwin-pilot-20261020/PROGRESS.md) |

## M17 — 四套既定评估：计算完成，审计未放行

6000完整episode，四套各T/C/P500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。[成功/长度/罚分及门](../../experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)。TF/PF2全部八组合P50/P90在旧界内，max全部超过1.30；首两S严格配对FAILED、训练/DEV曝光、组件审计与全部失败保留。无匹配500局MLP；不是独立TEST/非劣。GPU计算已清理，科学门仍未完成。

## M31 — 独立静态tile子图：completed

[原结果](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)：9block/72状态和独立54状态有限通过。不能继承给M34新算术。

## M32 — 阶段内persistent：completed

[原结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)：9block/90状态/48原语有限通过；378样本中B14比静态慢4.42%–6.26%，A/A最大10.07%。attention/FFN1阶段内常驻，其他GEMM仍host cuBLASLt；不满足整层单kernel目标。731block、77.927秒，owned退出。

## M33 — 退化profile：completed

[归因](../../experiments/single-block-persistent-profile-20261019/RESULT.md)：长B14两路径各34kernel；增量207.072us由三替换组件207.2425us解释。216 CTA反事实把短中长退化降到0.726%/0.474%/0.179%，仍无正收益。NCU权限失败、计数UNKNOWN，两次扩展身份实施失败保留。293block/81.856秒、owned退出。

## M34 — 整层单kernel与warp分工：completed，验收NO-GO

[主源码](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)与[入口说明](../../../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。全部self/cross、投影、FFN、Norm/RoPE/调制/残差/KV在一次cooperative launch内，8warp分调度/加载/计算/写回；粗阶段grid barrier，没有跨阶段细流水。9真实签名各trace1kernel、tile/输入/RNG/参数及metadata通过；9最终hidden逐位FAIL，相对L2约0.13%–0.56%，18写序列中9数值FAIL。

[两进程270样本](../../experiments/single-layer-warp-specialized-20261019/RESULT.md)：耗时为native的5.12–11.28倍；不是有效加速。172 registers/thread、59616B shared、理论1 CTA/SM；非实测occupancy。[阶段诊断](../../experiments/single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)B14 GEMM74.66%，唯一ldm40副本下降6.5%–19.0%，仅match失败v1、仍未准入。[精度边界](../../experiments/single-layer-warp-specialized-20261019/PRECISION-CUTS.md)独立定位Q、FFN2和video attention差异，不以splitK单因解释。

累计600/600block、0full30/episode，01:44:39北京清理。编译/依赖/CPU核对失败保留；本机88及远端86已部署保护项不变，远端未迁入state/rules的原88检查false不改。[资源/止损](../../experiments/single-layer-warp-specialized-20261019/RESOURCE-AND-STOP-REPORT.md)。不扩扫参、不集成、不降低门。

## M35 — 新同卡工程计时：已收口，原面板PARTIAL

[实际快照](../../experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)：330/360完整局，固定10task/init49×三臂×3独立server重复；计时重复不增加独立质量N。Goal/Spatial各90，Object60；Object1误送旧Goal/Spatial白名单在0episode/0teacher失败，保留不重试，只接续原未启动2/3，最大330/360。Long三phase已全部终态并清理；四套本轮没有继续排队的GPU作业。

GPU0 Goal→Object，GPU1 Spatial→Long；同case对照同卡，每卡单worker、server/client2线程。05:34:31停新、05:44:31清理（10/20），1452000尝试/3GiB/12 GPU生命周期小时帽不延期。首两S组件FAILED仍列出，计时只作未准入候选成本观察；不开放旧45chunk。warm含reset/policy/cache/env，不含env.close/落盘；冷setup另列。[边界](../../experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。

## M36 — RoboTwin只读调研：completed

[官方核查](../../experiments/robotwin-migration-survey-20261019/RESULT.md)：有LingBot-VA官方权重，50唯一任务，原适配100有效seed/task（先expert过滤），T/C/P全套15000局/condition。Q32/有效16动作通道、pool静态9792，当前TF64/API/阈值不能直接继承。没有装环境、下权重或启动实验。

## M37 — 数值契约与成本归因：completed，数值NO-GO

[成本归因](../../experiments/device-contract-20261020/TASK-COST-DIAGNOSIS.md)：Long全尝试1.107×、共同成功1.344×；同一task4/init49三次TF失败拉长任务，不能外推总体SR。Object共同成功/全体均1.216×，action1.575×、chunk1.341×，再受多3.4%chunk与推理外开销影响；优化reset不是主要瓶颈。共同成功只是条件指标，仍保留全attempt/SR/含失败成本。

[设备候选结果](../../experiments/device-contract-20261020/RESULT.md)：远端4项公共tile/stage/split宏配置通过，唯一CUTLASS候选Q/FFN2四项原逐位仍FAIL。重复/输入/参数/RNG均通过；不是根因证明。4参考block/12原语/8heuristic，0机器人局；首次0block DSO路径实施失败保留，独立修正未重置截止。生命周期185.698秒，owned活动0；本机88/远端86保护项不变。数值门未过，停止异步整层接入与扩扫；没有新GPU队列。[资源止损](../../experiments/device-contract-20261020/RESOURCE-AND-STOP-REPORT.md)。

## M38 — 当前整层退化profile：completed

[实测](../../experiments/whole-layer-bottleneck-20261020/RESULT.md)：6真实B2/B14短中长KV、222block/300原语、144完整计时。whole比同输入native慢5.09–11.14倍；原Torch/native6逐位通过、whole6失败。插桩副本GEMM61%–84%，短B14 cross K/V占63.3%；同输入Torch GEMM量级快约10倍。静态领取配对中位数慢3.1%–12.8%，无队列改动收益。profile/静态有限匹配失败v1，不是数值准入。

原实现1kernel，native39/35计算kernel；memory/注释单列。高寄存器/低理论occupancy不是单因，成熟GEMM同样一CTA驻留却快得多；同步单buffer和attention串行softmax有源码依据，硬件stall/流量UNKNOWN。[代码讲解](../../experiments/whole-layer-bottleneck-20261020/IMPLEMENTATION-GUIDE.md)。86.490秒生命周期、owned三个进程退出，两卡空，主源码/保护项不变，无新机器人任务。

## M39 — 单层异步tile与内部FA2：completed，整层数值NO-GO

[内部FA2结果](../../experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)：直接修改optimized，在整层单kernel内接入固定FA2 self/cross、split-KV/combine。9真实签名、18同输入attention全部逐位PASS；9重复/输入/RNG/参数、18相对旧async的有效KV转移和18单kernel trace通过。完整hidden仍9/9逐位FAIL，相对L2 0.133%–0.563%，最大0.5–1.5；同输入Q/FFN2独立误差和video短/中Norm2的12/8元素差异已定位，长video首次差异仅FFN2。

两个进程216样本：相对旧async，短action慢8%–9%，长action耗时降18%–21%，长video降43%；仍比native慢3.20–7.60倍。108 CTA、255 registers、82464B shared、2168B local资源，非实测spill/occupancy。387block＋18SDPA＋18Linear；GPU worker生命周期69.979秒，阶段到清理836.9秒，约6.79MiB。两次CPU编译失败保留，7组父子14个PID/birth退出，两卡空；本地75/远端71保护项不变。无新机器人/训练/30层，四false保持。

旧[M39双缓冲](../../experiments/single-layer-async-gemm-20261020/RESULT.md)675block、264.838秒、432样本及240/900登记不一致FAILED保留；旧预算未重置。本阶段沿既有记录承接用户FA2授权，没有新建实验登记。下一优先FFN2/Q及小量Norm2数值，再处理短Q资源压力与跨阶段就绪调度。

## 下一步及保护

M34–M39已终态；M37失败CUTLASS未集成，M39是保持旧WMMA数学的双缓冲，不等于M37数值修复；不能把普通cache收益叫协同创新。[claim矩阵](../../experiments/single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)、[日期](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。

四权限false；旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据、default A_off不改。baseline及依赖/投机源/原tile与HW/state/cycle/rules保护；仅optimized新实现，不重置/旧sync/升级，不建缺失cycle.yaml。远端只~/cxz，AutoDL不再SSH/关机UNKNOWN；不杀他人、不改服务/时钟。

M17旧10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB保留，不由M35重置。无新训练/120/独立TEST/第二环境/30层授权。仅main公开核对源码/HW/小证据，无模型/PT/KV/DSO/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文包；正文由用户飞书整理。之前状态已归档[history](history/M34-complete-20261020/PROGRESS.md)。

## 2026-10-20T12:38:31+08:00：新授权实际执行

用户同意两卡计划，已保存[执行决定](../../memory/global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)。M40仅在~/cxz内准备，原Target/采集/DEV尚0局；固定源码已迁入，Git传输失败保留，改用固定commit源码归档；国内pip镜像安装和固定资产/权重流式传输进行中。M39同输入6block/24原语控制确认三项主要差异与BF16归约路径有关；移植原Norm设备体后3个video的Norm2及其前序直到GELU逐位相同，整层仍FFN2差异，唯一serial split挑战者在编译。两次实施失败保留，未改旧门或原默认配置。

## 2026-10-20T13:42:42+08:00：O1有限数值门完成；两卡后继

[M39/O1](../../experiments/single-layer-async-gemm-20261020/NUMERICAL-ALIGNMENT-RESULT.md)9输出及18KV序列直接对原Torch逐位通过，261/300调用，旧失败不变；O2阶段测量后优化大M GEMM/Norm IO。M40权重/资产/私有运动规划就绪，首次专家场景OIDN非法访存，停止owned worker并记450.318秒，0policy/teacher；依官方资料私有兼容修复，不改变渲染配置，不重置18:20准备截止。

## 2026-10-20T13:56:55+08:00：O2收口、O3交接与M40止损

[O2](../../experiments/single-layer-async-gemm-20261020/PRIMITIVE-EFFICIENCY-RESULT.md)390/400 block，两独立进程216样本、18单kernel trace；数值通过，但仍慢native2.59–5.14倍。两轮同名扩展初测INVALID和两次PID命名空间检查失败均保留，不能用它们挑收益。O3首轮九输出/18KV通过，B14/video观察到72个消费者领取发生在其他producer全部完成计数之前；修正无界ticket空转后复验/计时，500帽不变。

[M40](../../experiments/robotwin-pilot-20261020/ENVIRONMENT-STOP-RESULT.md)原OIDN与私有2.3.3均首场景非法访存，495.321秒、0policy/0teacher；自有GPU链已清理，无自动后继。未关闭光追/去噪或改相机以绕过，原Target前置仍阻塞，后续训练/DEV未运行。
