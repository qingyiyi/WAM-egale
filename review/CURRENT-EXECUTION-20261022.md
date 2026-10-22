# 当前：FFN2写回与生产函数内联完成

A四元素写回/列常量外提，完整层同轮延迟降1.611%/1.349%/1.334%；B仅内联FFN1生产函数，同轮再降1.020%/1.156%/0.811%。两项各9真实/36结构有限逐位、输入/KV/RNG通过，计时与精度fixture一致。

当前原Torch/完整单层（远端A100-SXM4 GPU0、warm Graph、双方预计算Cross K/V）：**1.2185 / 1.7605 / 2.8473×**。当前独立FFN为 **0.9182 / 0.9167 / 0.9161×**，仍慢8.9%–9.2%。这不是E2E；不能宣布FFN已经追平。

生产函数独立调用消失，合计静态HMMA保留；每线程静态local1232→920B，动态流量未测。重放副本完整层扰动≤0.24%，资源一致。两轮改善百分比不直接相加，旧R2独立FFN约24%差距保留为历史。

两机研究入口采用B，16 owned身份退出，本机未GPU重测。累计12338.592/36000秒；原FAILED/截止、四项科学权限和production_admitted=false保留。下一步定位当前FFN1/FFN2计算和搬运差距；未启动新队列。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/RESULT.md)、[阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/STAGE-MAP.md)、[源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[HW](../.pipeline/PROGRESS.compact.md)。历史TF暂停72交接及RESULTS.json原样保留。
