---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
current: M39
next: M39
updated: '2026-10-20T23:19:23+08:00'
---
# 当前研究进度

[PLAN](PLAN.md)；current M39，next M39按profile修pointwise/layout，M40并行固定采集。

## 夜间接续 2026-10-21T01:11:44.030557+08:00

用户最新明确**先对齐再调度；无法对齐须给可追溯理由**。[修订](OVERNIGHT-PLAN-20261020.md)。ready_context_r2在切换优先级落实前完成，原结果保留，活动代码已撤回该调度路径。当前仅逐阶段顺序执行，baseline/投机生产代码不动。

GPU0：pointwise/layout/Norm搬运之后，RMS双行并行、B14 FFN1的M128 tile、cross-attention连续K/V原FA2加载分别通过9签名/23中间量/有效KV/RNG有限逐位，完整层分别改善约2%、1.3–1.6%、约1%。最新已完成cross_direct_load中/长约3.88/4.27ms，对照native2.40/2.84ms，**尚未对齐**。短KV存在跨进程时钟/热状态波动，全部轮次保留。当前table地址简化正在有限测量。

关键归因证据：原版FA2 128线程、64KiB shared；整层256线程、98848B shared、255 registers/thread，运行时驻留上限1 CTA/SM。阶段单独运行也有FFN1/cross-attention明显差距，不能全归因于grid.sync。编译spill/最终stack和普通kernel profile须分开，动态NCU counter不可用，不伪造带宽/occupancy实测。原[阶段矩阵](../../experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/STAGE-POST-RESULT.md)仍适用于三项基础优化后的版本，等待当前版本新矩阵。

GPU1：Target48已全部终态，41成功/7自然失败，训练数据FIT36/CAL12分别22785/7693行（排除各chunk首个无previous状态的更新）；PARENT2000最终权重与一次CAL已完成，非闭环质量证书。首接口因FSDP2动态类型识别FAILED，r2通过有限8policy对照；随后parent首局place_object_basket/valid08在0环境动作因重复terminal计数被拦住，PARTIAL保留，**不自动重试**。修正wrapper让S/T完整action路径互斥后r3八policy核验通过（S/T均精确一次terminal，S合法50更新，输出/有效KV/RNG/native-vs-shared树一致）。其他17个未开始case已按原截止接续，失效一局补跑请求待用户答复；不改变旧结果或把parent数据当DEV。

[迁移小证据](../../experiments/robotwin-pilot-20261020/overnight-remote-evidence/tf-migration-20261021/interface-gate-r3/RESULT.json)。完整episode reset bank等价仍未测，未启动TF/PF2成功率DEV。M39累计10GPU小时与本夜3小时帽、M40累计24小时/174局不重置，最迟09:30清理。四质量权限保持false。

## M39 简易版已选定

[结果](../../experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)。9真实签名输出/每格23中间量、18有效KV转移、重复、输入/RNG有限逐位通过，误差0。两进程18样本/engine/case交错测量，简易/复杂约0.9932–0.9959，未发现稳定退化。用户再次明确删除复杂活动路径，已归档旧FFN-ready与独立pipeline，后续仅维护简易whole_layer。

完整层简易/native两进程：短1.847–1.928，中1.8625–1.8628，长1.7126–1.7142。uncached/eager，输入/cache准备排除，远端GPU0无争用；不是强cache/Graph、30层、chunk/E2E。短KV跨进程时钟/热状态波动保留，不以最优轮替代全部记录。

[阶段矩阵](../../experiments/single-layer-async-gemm-20261020/stage-alignment/STAGE-RESULT.md)原版A/自有独立B/整层嵌入C：cross K/V独立约1.09–1.10倍；FFN1约1.83–1.84倍；cross-K RMS约3.4倍、GELU约6倍、table约5倍、layout约21–36倍。C包含队列/同步及不同缓存历史，插桩资源改变，不解释成纯算术比。原版self attention含pack。所有有效stage输出/KV有限逐位；NCU counter未知。

普通简易整层108CTA×256、255register、98848B shared、1624B local；GEMM4compute warp/8warp epilogue，pointwise仍4compute warp。阶段grid.sync和全局中间workspace仍在，未实现ready跨阶段调度。

原五项失败保留，包括第二进程比较16个无效slot导致FAILED；新诊断2144有效slot和数学输出一致，修正外部脚本另名补测。662/720block、4524/6000stage，本阶段106.745秒GPU生命周期、约741.608秒CPU准备构建、证据约59.3MiB。18owned根进程已回收，GPU0空。[终态](../../experiments/single-layer-async-gemm-20261020/stage-alignment/FINAL-AUDIT.json)。M39已核对账本累计约0.185GPU小时/10小时，原CPU/GPU口径分列；不重启过期旧链。

## M40 RoboTwin运行中

截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。原固定42自动队列只采Target，不是TF评测。合并目标Target48 FIT36/CAL12，复用108完整512-token BF16 bank；完整任务bank等价尚未测。GPU1单worker；截止10/21 03:37:02、含清理6小时及原24小时总帽不变。[现场](../../experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)。

## 其余状态与边界

四套LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。训练/DEV曝光、首两S审计FAILED、长尾保留，非独立TEST。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保持；原action沿原算子。baseline/投机/原tile/保护HW不改，BAAI只写~/cxz，AutoDL归还。不改服务/驱动/时钟、不杀他人。仅main发布源码/HW/小证据。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包。原[进度](history/simple-first-start-20261020/PROGRESS.md)保留。
