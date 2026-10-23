# 当前执行结果：GPU0已自然收尾，GPU1继续RoboTwin

2026-10-23T14:11:40.212610+08:00。GPU0于**14:01:34**结束中午窗口，新增18完整局/6组，T4/6、C3/6、P3/6成功，六组配对审计通过。owned出生身份核对全部退出，显存11MiB，共享lease闭合；无损归档及中午CPU watch终态。GPU1继续open_microwave-valid018-T，主5分钟CPU watch正常，无global halt。

RoboTwin已完整340/1800，113组完整三臂pair，另1局P成功所在组未齐。严格同113组：

|方法|成功|SR|
|---|---:|---:|
|Target|92/113|81.42%|
|TF连续|88/113|77.88%|
|TF+PF2|90/113|79.65%|

PF2相对C：6组救回、4组损失成功、84共同成功、19共同失败。全终态P91/114与配对P90/113分开。microwave配齐18组T4/C1/P1，当前失败较多，根因未确定。扩大DEV中途结果，不能外推最终SR、独立TEST或非劣。

原共享160GPU小时含活跃与准备已用26.033、余133.967；剩余1460局按当前task×arm时长粗估102–129单卡小时，约4–5.5天，非置信区间且不扩原预算。raw`PARTIAL_SHARED_BUDGET_NO_NEXT_PAIR`保留，本次由作业截止的下一完整pair预留触发，不是共享160小时耗尽，不是截断episode。GPU0现在释放，不自动开始新算子。

网页版已收到[现状与用户想法交接prompt](../.pipeline/experiments/single-layer-async-gemm-20261020/DAG-DESIGN-REVIEW-PROMPT-20261023.md)；用户要求其独立评议方向和优先级，不预设详细DAG设计。固定算子版本319ac6a17daed34e80692796f72a97bf44f4a99f，本轮算子未改动。

算子本轮结果保持：当前完整单层warm Graph相对原Torch **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。六投影8warp，QKV三套独立权重/输出共用288任务队列，grid同步27→23，数学tile3408/3480/3408。联合队列额外改善仅0.08%–0.44%，不作显著或应用收益结论。

三个可执行候选各9真实/36结构、225次有限完整调用逐位和状态核验通过，不证明所有输入。大M128 tile虽精度通过却慢9%–11%，已回退；K128固定库布局CPU编译不支持，0GPU调用，失败保留。完整新版阶段表已重测，QKV联合原Torch/自有约0.86×，Self-out约0.71–0.80×、Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN约1.01–1.02×；不能拆假计时或相加分账。

双方同等最终Cross K/RMS、V预计算；原Torch使用静态dense history。这不是30层、chunk、机器人E2E，也不是最强guarded实现增量。源fb650be4…、DSOa41d778d…保持，M39累计18427.875/36000秒，余4.881小时；无新算子GPU作业/远端NCU。后续可讨论保持BF16边界的部分和私有暂存，尚未实现。

[中午窗口结果](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/RESULT.md)、[清理与全局统计](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/AFTERNOON-CLOSE-20261023.json)、[GPU1健康](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/AFTERNOON-HEALTH-20261023.json)、[新版完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[HW入口](../.pipeline/PROGRESS.compact.md)。四科学权限与production_admitted均false，旧FAILED与review/RESULTS.json不改。[下午更新前入口](history/current-before-afternoon-close-20261023/CURRENT-EXECUTION-20261023.md)。
