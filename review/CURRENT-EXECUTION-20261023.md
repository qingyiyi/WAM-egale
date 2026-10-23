# 当前执行结果：算子本轮完成，RoboTwin上午停派核验

2026-10-23T10:09:05.210887+08:00。RoboTwin完整终态**256/1800**，85完整三臂pair审计通过，另1局PF2已终态但所在pair未齐。GPU0完成150局/50组，于**10:02:27**自然收尾，PID出生身份核对无owned残留，显存11MiB；GPU1继续原固定队列。无global halt，5分钟CPU watcher运行。

|同一85组完整DEV|成功|SR|
|---|---:|---:|
|Target|69/85|81.18%|
|TF连续|66/85|77.65%|
|TF+PF2|70/85|82.35%|

PF2相对连续TF：6组救回、2组损失成功、64组双方成功、13组双方失败。全终态P70/86与配对P70/85分开；尚非独立TEST或最终总体SR。M40含活跃累计20.075/160 GPU小时，余139.925小时；剩余1544局根据现有task×arm时长外推单GPU约4.5–6天，非置信区间，未知长尾与准备可增加，不扩预算。完整reset bank和COMBINED沿用；GPU0释放后未自动开新算子。

算子本轮结果保持：当前完整单层warm Graph相对原Torch **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。六投影8warp，QKV三套独立权重/输出共用288任务队列，grid同步27→23，数学tile3408/3480/3408。联合队列额外改善仅0.08%–0.44%，不作显著或应用收益结论。

三个可执行候选各9真实/36结构、225次有限完整调用逐位和状态核验通过，不证明所有输入。大M128 tile虽精度通过却慢9%–11%，已回退；K128固定库布局CPU编译不支持，0GPU调用，失败保留。完整新版阶段表已重测，QKV联合原Torch/自有约0.86×，Self-out约0.71–0.80×、Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN约1.01–1.02×；不能拆假计时或相加分账。

双方同等最终Cross K/RMS、V预计算；原Torch使用静态dense history。这不是30层、chunk、机器人E2E，也不是最强guarded实现增量。源fb650be4…、DSOa41d778d…保持，M39累计18427.875/36000秒，余4.881小时；无新算子GPU作业/远端NCU。后续可讨论保持BF16边界的部分和私有暂存，尚未实现。

[上午结果](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/MORNING-RESULT-20261023.md)、[实际检查点](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)、[GPU0清理](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/GPU0-MORNING-CLEANUP-20261023.json)、[算子完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESULT.md)、[新版完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[HW入口](../.pipeline/PROGRESS.compact.md)。四科学权限与production_admitted均false；旧review/RESULTS.json、FAILED、预算起点/截止和保护文件未改。[上午更新前入口](history/current-before-morning-status-20261023/CURRENT-EXECUTION-20261023.md)。
