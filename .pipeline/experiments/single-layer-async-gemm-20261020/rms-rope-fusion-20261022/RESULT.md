# RMS与RoPE融合：有限精度通过，完整层小幅改善

更新2026-10-22T15:31:11.420822+08:00。仅在现有按需调制版的B14候选self Q/K路径中融合RMS与RoPE；GEMM、cross-attention数学及阶段调度不变，B2保留兼容分支。两机optimized采用实测候选，baseline/投机生产/原tile未改。

## 已观察

|KV|修改前ms|融合后ms|延迟下降|同轮guarded ms|仍慢于guarded|
|---|---:|---:|---:|---:|---:|
|short|2.270464|2.261504|0.395%|2.068736|9.32%|
|middle|2.434816|2.424576|0.421%|2.185728|10.93%|
|long|2.449152|2.438400|0.439%|2.199552|10.86%|


远端GPU0 A100-SXM4-40GB，普通debug=false、uncached context完整单层CUDA Graph。两独立进程，各五臂预热600次、24轮交错；表为两个进程中位数的算术平均，各进程9408完整层调用。准入和case边界无外部GPU进程，SM1410前后核对。全部六个进程×KV对照同向改善，收益约9–11微秒/层，幅度小，不声称统计显著或重大加速。最强同轮guarded仍快9.32%–10.93%。不含Python分配，也不是30层/机器人/E2E。

9真实签名81次、21结构84次完整层调用，输出及23共有中间量逐位相同max_abs=0；有效KV、输入/RNG、重复、普通/debug及B2合法cache更新序列通过。仅有限fixture范围。三个B14计时fixture与精度使用同一输入，计时现场还核对输出/状态。

每个4warp组处理一行3072维RMS，两个组/CTA。保持原vec4归约树、gamma乘法、RMS输出BF16舍入，再使用原FP64 RoPE与最终转换；归约完成后直接在寄存器中旋转相邻元素。debug仍保存归一化中间值用于核验，normal不物化SQN/SKN。

B14普通路径少两个BF16中间张量，共2.625MiB；两处独立RoPE阶段取消，逻辑tile11912/11984→11576/11648，全grid同步38→36。debug因MOD多一个阶段，为37次。上述为源码/分配推导，不等于实测HBM或峰值显存减少。仍是一层一个persistent kernel，没有恢复ready队列或新增GEMM-epilogue流水。

资源为108CTA、256线程、100096B shared、255寄存器/线程、static localSize936B；上轮928B。静态local字段不是动态spill流量，不能由这8B变化归因全部性能。

## 身份与证据

源whole_layer.cu `0f79579d26d29dd9b9532d8d72824eed177b64d82b0e58b8add078f6063d844b`；候选DSO `795be814c40a152583e13a34f6cf6bb5510b1fc38f5b4cd8c26076b4dc4efd75`，修改前DSO `41af9a655a2387e4871eb5ab5dee1a9ac0cf613fd35c66f1f076fc5ef0f20b41`保留在上一轮。两机部署见资源报告；四科学权限false。

[差异](CHANGE.patch)、[9签名](SIMPLE-GATE-RESULT.json)、[21结构](STRUCTURAL-RESULT.json)、[原始计时](WHOLE-LAYER-TIMING.csv)、[汇总](GPU-SUMMARY.json)、[阶段表](CANDIDATE-STAGE-MAP.md)、[资源与清理](RESOURCE-AND-STOP-REPORT.md)。CPU编译308.493秒、整窗153.757秒；没有追加模型或机器人测试。

[Tensor Core与RoPE核验](TENSOR-CORE-AND-ROPE.md)：活动before二进制的投影与attention函数有HMMA，QK/PV源码均实例化BF16 Tensor Core MMA；候选全二进制也含HMMA，见[SASS静态分类](SASS-INSTRUCTION-KINDS.json)。这不是新动态利用率profile。

[MegaMoE角色解释](MEGAMOE-WARP-NOTES.md)依据固定SM100官方实现；本项目SM80的当前投影仍先MMA后协作写回，没有实现跨tile epilogue重叠。

## 下一步

本轮收口；继续按用户逐阶段讨论。投影+RMS涉及24列块完整行归约，跨tile GEMM/epilogue流水需独立缓冲与同步设计，均未实施。小幅融合收益不能替代仍缺少的强基线增量、有效提交成本和闭环证据。
