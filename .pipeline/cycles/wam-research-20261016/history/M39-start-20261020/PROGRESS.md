---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-20T09:57:08.243204+08:00'
current: M38
next: M39
---
# 当前研究进度

更新：2026-10-20T09:57:08.243204+08:00。当前范围只有一层；原四套计算和新计时重复分开。[短入口](../../PROGRESS.compact.md)。

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

## M39 — GEMM/attention设备tile数据通路：pending

建议先做真实M/N/K的SM80加载/计算流水与数值门，再考虑更复杂跨阶段ready调度。只一层、仅optimized；下一个具体作业尚未登记/启动，M38余额不延续。不是恢复数值扫参或已证明MegaTransformer收益。

## 下一步及保护

M34/M35/M37有界作业已终态；M37数值门失败，后继整层异步集成未触发，不重复旧网格搜索；不能把普通cache收益叫协同创新。[claim矩阵](../../experiments/single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)、[日期](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。

四权限false；旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据、default A_off不改。baseline及依赖/投机源/原tile与HW/state/cycle/rules保护；仅optimized新实现，不重置/旧sync/升级，不建缺失cycle.yaml。远端只~/cxz，AutoDL不再SSH/关机UNKNOWN；不杀他人、不改服务/时钟。

M17旧10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB保留，不由M35重置。无新训练/120/独立TEST/第二环境/30层授权。仅main公开核对源码/HW/小证据，无模型/PT/KV/DSO/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文包；正文由用户飞书整理。之前状态已归档[history](history/M34-complete-20261020/PROGRESS.md)。
