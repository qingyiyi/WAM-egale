# 当前执行结果：六投影8warp、QKV联合队列与临时双卡

2026-10-23T02:17:22.010612+08:00。算子已在两机研究入口采用，GPU0已真实交接RoboTwin。

最终同轮四臂比较当前原Torch/自有完整单层warm Graph **1.2967/1.8469/2.9910×**（KV288/1152/2160）；相对用户上版延迟降2.211%/1.995%/1.841%。其余三投影改8warp是主要收益，联合QKV增量仅0.08%–0.44%，不称显著或应用收益。三套Q/K/V权重和输出独立，288任务共享队列，grid27→23；数学tile3408/3480/3408不变。

三个可执行候选各9真实/36结构、225次有限门通过。大M128输出tile精度通过但慢9%–11%，回退；K128默认固定库布局编译拒绝，0GPU调用，保留FAILED。当前完整阶段表全部重测，QKV联合约0.86×，不拆假单项计时；Self-out约0.71–0.80×、Cross-Q约0.80–0.82×、Cross-out约0.74×，FFN约1.01–1.02×。阶段缓存/计时边界不同，不相加分账。

双方同最终Cross K/RMS和V预计算，Torch静态dense history；不是30层、chunk、机器人E2E或最强guarded增量。远端无counter权限，NCU0；保持同一per-tile K1536/BF16/bias边界。未来部分和CTA私有暂存仅建议未实施，本轮不再启动算子GPU。

源码fb650be4…、DSOa41d778d…，02:03全部算子owned退出，累计18427.875/36000秒。GPU0/1各一个RoboTwin worker，原27局在完整pair后PAUSED保留，新原子队列去重；每pair三臂同卡，原160GPU小时/10天、T0/失败记录不改。GPU0以10/23约10:00为目标预测完整pair时长停派，已开局自然终态；GPU1继续。完整512 BF16 bank（480条）直接引用，不重复编码/保存大文件。

截至2026-10-23T02:12:38.420042+08:00，合计37/1800，T11/12、C11/12、P11/13；12完整pair审计通过，GPU0首组3/3成功并初态/首两chunk通过。当前扩大DEV早期状态，不外推总体SR或独立TEST。两卡5分钟CPU只读巡检运行，无自动重试。机器人仍使用冻结评测配置，不能继承新算子性能或质量结论。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESULT.md)、[新版完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[身份](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/SELECTED.json)、[算子资源](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESOURCE-AND-STOP-REPORT.json)、[双卡计划](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/PLAN.md)、[实际检查点](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)、[HW入口](../.pipeline/PROGRESS.compact.md)。四科学权限及production_admitted=false；旧review/RESULTS.json未改。
