# 当前执行结果：GPU0中午接续至约14:00，DAG设计交网页版

2026-10-23T12:13:08.831264+08:00。用户已另行授权GPU0中午从新namespace接续RoboTwin固定队列。已真实执行place_object_basket-valid016-T，teacher returned1262、环境命令464，初态物理/观测exact。GPU1继续open_microwave-valid015-P，两卡各一个worker；没有算子GPU作业。

截至本检查点完整285/1800、95完整三臂pair，同95组T77/95、C74/95、P77/95。无global halt；这是扩大DEV中途状态，不能外推最终SR/独立TEST。两卡低开销watch运行，原数学/server/client源未改，完整reset bank/COMBINED复用。

GPU0预计今天14:00附近自然收尾：按实测task/arm P90估整pair耗时，13:15后不接预计超过14:15的组、14:00后不接新组；已接完整局沿原终态和1800秒上限，不人为截断。原160GPU小时/10天/T0、失败保留；当前含活跃与准备累计22.251小时。窗口结束后不自动启动算子。

用户将QKV交错派发留到完整单层依赖图设计，静态/动态/混合调度尚未定案。请网页版使用[完整DAG设计prompt](../.pipeline/experiments/single-layer-async-gemm-20261020/DAG-DESIGN-REVIEW-PROMPT-20261023.md)，固定319ac6a17daed34e80692796f72a97bf44f4a99f读取算子源码；本次未改算子。

算子本轮结果保持：当前完整单层warm Graph相对原Torch **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。六投影8warp，QKV三套独立权重/输出共用288任务队列，grid同步27→23，数学tile3408/3480/3408。联合队列额外改善仅0.08%–0.44%，不作显著或应用收益结论。

三个可执行候选各9真实/36结构、225次有限完整调用逐位和状态核验通过，不证明所有输入。大M128 tile虽精度通过却慢9%–11%，已回退；K128固定库布局CPU编译不支持，0GPU调用，失败保留。完整新版阶段表已重测，QKV联合原Torch/自有约0.86×，Self-out约0.71–0.80×、Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN约1.01–1.02×；不能拆假计时或相加分账。

双方同等最终Cross K/RMS、V预计算；原Torch使用静态dense history。这不是30层、chunk、机器人E2E，也不是最强guarded实现增量。源fb650be4…、DSOa41d778d…保持，M39累计18427.875/36000秒，余4.881小时；无新算子GPU作业/远端NCU。后续可讨论保持BF16边界的部分和私有暂存，尚未实现。

[GPU0中午计划](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/PLAN.md)、[实际启动收据](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/START-RECEIPT.json)、[检查点](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/CURRENT-CHECKPOINT.json)、[新版完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[HW入口](../.pipeline/PROGRESS.compact.md)。四科学权限与production_admitted均false；旧review/RESULTS.json不改。[中午更新前入口](history/current-before-midday-allocation-20261023/CURRENT-EXECUTION-20261023.md)。
