# 当前：FFN1尾部行块分摊已采用

最后4个FFN1生产tile各拆3个M32任务，共112→120；FFN2的96个任务及原归约/舍入不改。9真实签名、36结构控制有限逐位/状态通过；计时与精度fixture一致。

原Torch/当前完整单层（warm Graph，双方预计算Cross K/V）：短/中/长为 **1.1910 / 1.7241 / 2.7993×**。本次同轮R2内部回归控制显示完整层延迟下降1.932%/1.918%/1.856%；两独立进程每档均改善。不是机器人E2E，也没有重测当前独立FFN差距。

debug支持最后组更早就绪：相对首组的间隔约118–119→89–90us；最后消费完成约246→224–225us。最后生产后的消费尾段反而略增，static local1144→1224B；下一步处理FFN2写回/索引和函数保存恢复。本轮没有新动态硬件计数。

两机研究入口同步，远端owned退出，本机未GPU重测。累计11295.646/36000秒，原截止/FAILED保留；四项科学权限及production_admitted仍false。LIBERO/RoboTwin本轮未新增作业；原paused72和review/RESULTS.json未改。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/RESULT.md)、[阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/STAGE-MAP.md)、[源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[HW](../.pipeline/PROGRESS.compact.md)。
