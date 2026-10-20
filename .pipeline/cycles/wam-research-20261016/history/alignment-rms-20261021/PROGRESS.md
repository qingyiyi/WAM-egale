---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
status: active
updated: '2026-10-21T01:45:15.402937+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 结果/证据 | 下一步 |
|---|---|---|---|---|
| `M17` | LIBERO | in_progress | 6000局已齐，旧审计未放行 | 保留曝光、首两S失败、长尾 |
| `M39` | 单层对齐 | in_progress | 最新单层有限逐位通过、速度未对齐 | 逐阶段修复，暂停新增调度 |
| `M40` | RoboTwin | in_progress | Target48与parent训练齐，剩余17采集中 | 一局补跑等待用户答复 |

## M39 当前事实

用户最新要求**先对齐再调度；无法对齐给出证据理由**。baseline/投机/原tile不动；仅单层、仅optimized简易版本。ready_context_r2已结束证据保留，活动源码撤回；当前仍逐阶段grid.sync。

最新已选版本：pointwise/layout/Norm搬运、cross-K RMS双行、FFN1 M128N128、cross连续KV原FA2加载、B14 Q16 M64 attention。两独立进程完整层短/中/长约3.94/3.60/3.85ms，对应native约2.63/2.40/2.84ms；约1.50/1.50/1.36倍，**未速度对齐**。短KV受频率状态影响，其他实测约1.56倍；不选最好轮。[原始因素证据](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/attention_m64/JOB.json)。所有9真实签名的输出、23中间量、18有效KV转移、输入/RNG/重复有限逐位误差0；不外推任意输入或完整机器人。

[当前阶段矩阵](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/stage_final/RESULT.md)：中KV原版A/自有独立B，cross K/V约1.07、cross attention1.11、FFN1 1.20、FFN2 1.40、cross-K RMS2.26、table2.5、GELU3.4倍；self-attention完整阶段约0.78倍（原版包括历史pack，不能只称FA2核心加速）。嵌入C包含队列/同步及不同缓存/编译资源，不能用C−B当纯barrier开销。

当前整层256线程、255register/thread、98848B shared，运行时上限1CTA/SM；原FA2 128线程/64KiB shared。这是静态容量与运行时资源证据，动态NCU流量/occupancy仍UNKNOWN。扩大FFN1 N256虽有限精度通过，却使整层慢约3–4%，已撤回；shared升至148000B，尚未单独证明L1影响。grid_constant原已实现，取消重复方案，不声称新增收益。point8192正在有限测量，尚未选定。

当前范围是远端GPU0同输入uncached/eager完整单层，准备排除、profile分离；不是最强cache/Graph、30层/chunk/E2E。M39累计10小时/本夜3小时生命周期帽不变，新增调用帽已前瞻记录5600block/12000stage。

## M40 当前事实

Target48全部终态41成功7自然失败，属于FIT/CAL采集，不是DEV成功率。Target FIT36/CAL12分别22785/7693行，已核query/CFG监督，排除chunk首个无previous行。PARENT2000、固定normalizer、一次CAL完成；RoboTwin tau0.007879490032792091不改LIBERO阈值。

原FSDP2动态类型门FAILED；修正后r2有限8policy通过，但没有抓到重复terminal。parent首局place_object_basket/valid08在0环境动作因重复terminal被拦住，PARTIAL保留。wrapper完整S/T互斥修正后r3八policy通过：精确一次terminal、S合法50更新、输出/有效KV/RNG/native-shared树一致。[小证据](../../experiments/robotwin-pilot-20261020/overnight-remote-evidence/tf-migration-20261021/interface-gate-r3/RESULT.json)。原失效局forward原件仅下限25/上限49，计预算按49，不补造精确数。

01:43剩余17已完成10局，当前open_microwave/valid09；原03:55截止不变。缺失valid08补跑请求待明确答复，不自动重试、不跳过固定FIT缺口。current18/最终ON/三臂DEV90尚未启动。bank为108完整512-token BF16条目，完整episode bank等价尚未测。M40累计24GPU生命周期小时/174局/1393500teacher帽不变，最迟09:30清理。

## 共同边界与下一步

LIBERO T/C/P各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499；不混作独立TEST。engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保留。

下一步：完成point8192门与两进程比较，保留负结果；解释剩余GEMM/Norm资源差距，不启动ready。定期巡检GPU1，不补跑缺失局。仅main发布核对源码/HW/小证据，无权重/凭据。10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00证据包。[此前完整进度](history/alignment-m64-20261021/PROGRESS.md)。
