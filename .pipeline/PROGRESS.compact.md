---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-23T02:23:30.193965+08:00'
current: M40
next: 巡检双卡RoboTwin与GPU0约10点自然停派；下一轮算子先讨论BF16部分和私有暂存，本轮不再启动。
---

# 当前研究进度（2026-10-23T02:23:30.193965+08:00）

M39本轮已完成并交接GPU0。当前采用六个同形状投影8warp＋Self QKV联合288任务队列；CTA64×128、warp32×32、K64/3stage、两段K1536和BF16/bias边界不变。数学tile3408/3480/3408，grid同步27→23。两机源码fb650be4…、DSOa41d778d…，原始action/B2路径保持原契约。

最终同轮四臂原Torch/上版/六投影版/联合版：当前完整单层warm Graph相对原Torch **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。主要收益来自其余三投影改8warp，联合QKV增量仅0.08%–0.44%（逐进程/档），小信号，不称统计显著或应用收益。不跨轮相乘。

三个可执行候选各9真实/36结构/225次有限完整调用逐位及状态通过；不证明所有输入。较大128×128 tile虽通过门，整层退化9%–11%，不采用；任务96→48、函数静态栈8→192B均有记录，动态原因占比未知。K128被固定CUTLASS默认Crosswise布局编译拒绝，0GPU调用，不当性能负结果。旧FAILED/负结果保留。

新版完整阶段表已全部两进程重测：QKV联合诊断原Torch/自有约0.86×，Self-out约0.71–0.80×，Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN共同区间约1.01–1.02×。QKV必须联合计时，不能拆三行或同旧单投影热缓存直接比较；阶段缓存/同步边界不同，不能相加分账。双方同Cross最终K/RMS与V预计算、Torch静态dense history；不是30层/chunk/E2E，也不代表最强guarded增量。108CTA×8warp、112384B shared、255寄存器、静态local1016B，测量副本资源同、扰动≤0.197%。

M39新增GPU workers250.776秒、CPU编译621.775秒；保守含准备/等待/清理1755.288秒，累计 **18427.875/36000秒**，剩余4.881小时。02:03:36所有owned退出、两机采用并向RoboTwin交接；02:34原作业截止保留，不再新增算子GPU作业。本机0GPU、远端0NCU。四科学权限及production_admitted仍false。

M40用户新批准临时双卡：原GPU1在27局/9完整pair边界PAUSED保留，新原子队列从剩余任务去重接续；同pair三臂始终同卡，总160GPU小时/10天、原T0和旧24小时历史不改。GPU0借用至北京时间10/23约10:00：以各任务/arm实测P90预测整组耗时，09:15后不接预计超过10:15的新组，10:00后不接新组；已开始的完整episode自然结束，原1800秒/局硬上限仍在。GPU1继续，GPU0释放后不自动开新算子。

截至2026-10-23T02:23:30.193965+08:00合计 **41/1800局**，T12/14、C12/13、P12/14；13完整pair审计通过，GPU0已通过2组。当前GPU0=handover_block-valid002-P，GPU1=open_microwave-valid001-C；无global halt。当前为扩大DEV早期状态，不外推总体SR；完整reset bank直接复用、5分钟watch继续，无自动重试。

|ID|阶段|状态|当前证据|下一步|
|---|---|---|---|---|
|`M39`|单层算子|in_progress|六投影8warp＋QKV联合队列，两机采用及新版全表|讨论投影BF16分段部分和的CTA私有暂存；尚未实现，不在当前GPU0继续跑|
|`M17`|LIBERO证据|blocked|四套6000局齐，独立性/配对/长尾缺口保留|不恢复过期队列|
|`M40`|RoboTwin扩大DEV|in_progress|临时双卡，41/1800，首GPU0完整pair通过|低开销巡检，GPU0约10点自然收尾，GPU1继续|

[本轮结果](experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESULT.md)、[新版完整阶段表](experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[算子资源](experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESOURCE-AND-STOP-REPORT.json)、[双卡计划](experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/PLAN.md)、[实际检查点](experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)。10/23冻结身份与claim、10/28主表、11/01 18:00证据包；证据不足收缩claim。
