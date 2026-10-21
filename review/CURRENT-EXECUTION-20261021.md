# 10/21 12:26：Cross K/V与FFN两项完成

固定本文件所在提交审阅。当前代码保留两项改动：Cross K/V整个阶段进入专用设备函数；B14 FFN固定形状展开。tile、CUTLASS算术、BF16分段、阶段同步不变。仍一个完整单层kernel，没有新ready调度。

|B14历史KV|native ms（两进程范围）|最终 ms（两进程范围）|比本轮前下降|仍比native慢|
|---|---|---|---|---|
|288|2.1197–2.1207|2.7663–2.7668|10.84%–10.87%|30.44%–30.53%|
|1152|2.4051–2.4052|3.0268–3.0285|9.99%–10.13%|25.85%–25.92%|
|2160|2.8441–2.8461|3.2809–3.2809|9.46%–9.59%|15.28%–15.36%|

这是直接最初版/最终版的未插桩完整层比较，固定60预热/交错12样本，GPU0无其他compute进程，GPU1机器人同时运行；uncached/eager，不是最强cached/Graph或机器人E2E。Cross单独增量8.84%–10.60%，FFN再改善0.587%–0.700%；不把分轮收益相乘。

两阶段各9真实签名、每格23中间量、18有效KV状态转移、输入/RNG有限逐位检查通过，最终输出max_abs=0。不是任意输入或闭环质量保证。

阶段时间戳显示Cross约792/787→609/607µs；FFN1约236→214µs、FFN2约167→159µs。插桩影响约±1%，不把阶段差相加当未插桩总收益。SASS支持旧每tile进入通用函数、新每阶段进入一次带来函数保存/恢复摊薄；动态spill、HBM/L2与achieved occupancy仍UNKNOWN，不能写成流量减半或占用率提高。

[完整报告](../.pipeline/experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md) / [全部计时](../.pipeline/experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/FULL-LAYER-ALL.csv) / [活动源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu) / [代码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。前次小算子与CTA等待记录见[历史检查点](history/gemm-function-control-20261021/CURRENT-EXECUTION-20261021.md)。

下一步先FFN主循环/分段归约/epilogue和整层资源差距，再最小DAG调度。每阶段按合适tile和自然任务数，108是worker池，不要求凑满。此前FFN2并行K整层负收益保留，未恢复。

本轮GPU worker生命周期120.995秒，CPU准备523.226秒，M39累计1770.668/36000秒，GPU0本轮进程已退出；CUDA active时间未测。GPU1 CURRENT18采集12:26:32为12/18完整、9成功，valid13下一局已运行；这不是DEV成功率。自动接审计/ON/CAL，DEV90尚未排队。

旧paused72、FAILED/retention、baseline/原tile/投机生产代码、四项false保持。公开快照不含模型权重、原始PT/KV/相机、DSO和完整profiler/SASS；不得声称仅凭GitHub已重算全部真实tensor。
