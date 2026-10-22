# 最终版本本机NCU

状态：OBSERVED_FINAL_VERSION_COUNTERS；guard=COMPLETE，worker=COMPLETE_NCU_FINITE_REAL_INPUT。

真实middle/B14/action/layer0，冻结DSOac3f9e46；仅诊断，不用于正式延迟表。先前0kernel外部干扰记录保持PARTIAL。

OBSERVED：四个NVTX范围完成；每个实际kernel分别列出，不将多kernel的百分比直接相加。

|kernel|grid/block|Tensor active %|active warp %|DRAM bytes|L2 sectors|local load+store bytes|barrier stall %|
|---|---|---:|---:|---:|---:|---:|---:|
|measure_whole_final/wam_whole_layer_v1::whole_layer(wam_whole_layer_v1::Params)|(108, 1, 1) / (256, 1, 1)|47.93|12.5|1422800128.0|263647958.0|243623424.0|40.67|
|measure_native_selfq/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn|(24, 2, 2) / (128, 1, 1)|61.92|6.19|17084160.0|3253877.0|0.0|0.47|
|measure_native_ffn1/ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn|(56, 2, 1) / (256, 1, 1)|83.71|12.49|92897792.0|11962888.0|0.0|7.03|
|measure_native_ffn2/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn|(24, 2, 2) / (128, 1, 1)|80.8|6.23|98931328.0|15771628.0|0.0|0.17|

HYPOTHESIS：local状态保存/恢复、固定资源配置和执行几何可能共同限制整层；这些counter不能单独确定每项耗时。

RECOMMENDATION：结合三KV逐阶段和资源干预解释剩余差距。没有完整native层counter，不宣称整层流量相对native下降或增加某倍数；不使用NCU时间生成speedup。
