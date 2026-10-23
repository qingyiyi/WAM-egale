---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-23T14:11:40.212610+08:00'
current: M40
next: GPU1继续固定RoboTwin队列与5分钟巡检；GPU0中午窗口已闭合清理，等待网页版基于现状与用户想法独立评议算子下一步。
---

# 当前研究进度（2026-10-23T14:11:40.212610+08:00）

M39本轮已完成并交接GPU0。当前采用六个同形状投影8warp＋Self QKV联合288任务队列；CTA64×128、warp32×32、K64/3stage、两段K1536和BF16/bias边界不变。数学tile3408/3480/3408，grid同步27→23。两机源码fb650be4…、DSOa41d778d…，原始action/B2路径保持原契约。

最终同轮四臂原Torch/上版/六投影版/联合版：当前完整单层warm Graph相对原Torch **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。主要收益来自其余三投影改8warp，联合QKV增量仅0.08%–0.44%（逐进程/档），小信号，不称统计显著或应用收益。不跨轮相乘。

三个可执行候选各9真实/36结构/225次有限完整调用逐位及状态通过；不证明所有输入。较大128×128 tile虽通过门，整层退化9%–11%，不采用；任务96→48、函数静态栈8→192B均有记录，动态原因占比未知。K128被固定CUTLASS默认Crosswise布局编译拒绝，0GPU调用，不当性能负结果。旧FAILED/负结果保留。

新版完整阶段表已全部两进程重测：QKV联合诊断原Torch/自有约0.86×，Self-out约0.71–0.80×，Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN共同区间约1.01–1.02×。QKV必须联合计时，不能拆三行或同旧单投影热缓存直接比较；阶段缓存/同步边界不同，不能相加分账。双方同Cross最终K/RMS与V预计算、Torch静态dense history；不是30层/chunk/E2E，也不代表最强guarded增量。108CTA×8warp、112384B shared、255寄存器、静态local1016B，测量副本资源同、扰动≤0.197%。

M39新增GPU workers250.776秒、CPU编译621.775秒；保守含准备/等待/清理1755.288秒，累计 **18427.875/36000秒**，剩余4.881小时。02:03:36所有owned退出、两机采用并向RoboTwin交接；02:34原作业截止保留，不再新增算子GPU作业。本机0GPU、远端0NCU。四科学权限及production_admitted仍false。

M40上午GPU0完成150局/50组，中午用户另授至约14点的新窗口现也闭合：**14:01:34**完成18局/6组后停派、owned出生身份核对全部退出、显存11MiB、共享lease关闭，归档与中午watch均终态。原raw状态`PARTIAL_SHARED_BUDGET_NO_NEXT_PAIR`保留；实际是本作业截止无法预留下一完整pair，并非原160小时耗尽。18局都完整自然终态、6组审计通过，无基础设施异常。GPU1继续open_microwave-valid018-T、主5分钟watch正常；原160小时/10天/T0/失败不改，不自动恢复算子。

截至2026-10-23T14:11:40.212610+08:00，合计 **340/1800局**、113完整pair；严格同113组T92/113、C88/113、P90/113，额外1局P成功所在pair未齐（全终态P91/114）。P相对C救回6、损失成功4，扩大DEV尚不能外推总体质量或非劣。含活跃与准备累计26.033/160小时，余133.967；剩余1460局按现有分任务/arm时长约102–129单卡小时（4–5.5天），非区间保证且不扩原预算。完整reset bank/COMBINED继续使用，四科学权限false。

用户将QKV交错派发留到完整DAG设计，静态/动态/混合尚未定案；网页版prompt已按用户澄清改为现状、实测限制与用户想法的中立交接，固定算子commit，方向和优先级由网页版独立判断；不预设必须展开详细DAG设计。

|ID|阶段|状态|当前证据|下一步|
|---|---|---|---|---|
|`M39`|单层算子|in_progress|六投影8warp＋QKV联合队列，两机采用及新版全表|讨论投影开销；QKV交错留到统一DAG设计，具体调度未定，不自动开新GPU作业|
|`M17`|LIBERO证据|blocked|四套6000局齐，独立性/配对/长尾缺口保留|不恢复过期队列|
|`M40`|RoboTwin扩大DEV|in_progress|340/1800、113完整pair；GPU0中午已闭合清理|GPU1继续与巡检；原160小时不增|

[本轮结果](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESULT.md)、[新版完整阶段表](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[算子资源](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESOURCE-AND-STOP-REPORT.json)、[双卡计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/PLAN.md)、[上午结果](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/MORNING-RESULT-20261023.md)、[实际检查点](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)。10/23冻结身份与claim、10/28主表、11/01 18:00证据包；证据不足收缩claim。


[中午GPU0结果](../../experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/RESULT.md)、[实际启动](../../experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/START-RECEIPT.json)、[网页版DAG设计prompt](../../experiments/single-layer-async-gemm-20261020/DAG-DESIGN-REVIEW-PROMPT-20261023.md)。
