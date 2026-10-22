# 当前单层Transformer：按需调制与RMS+RoPE融合

当前optimized是阶段同步版，已保留按需调制，并融合B14 self Q/K的RMS+RoPE。活动DSO `795be814c40a152583e13a34f6cf6bb5510b1fc38f5b4cd8c26076b4dc4efd75`。一个persistent完整层kernel，108CTA/256线程；没有ready队列或跨tile独立epilogue流水。

[阶段表](STAGE-MAP.md)列公式、分块、同步。`csrc/whole_layer.cu`的rms_pair_tile保留完整3072维归约，norm_rope在BF16归一化结果上执行原FP64旋转；normal不物化SQN/SKN，debug保存用于审计。cross Q/K只有RMS，V没有RMS/RoPE；B2保持原兼容分支。

短/中/长同轮完整层2.270/2.435/2.449ms→2.262/2.425/2.438ms，降0.395%/0.421%/0.439%，仍慢同轮guarded9.32%/10.93%/10.86%。9签名及21结构有限逐位/状态通过。[证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)。不是模型或机器人E2E。

B14普通逻辑tile11576/11648、grid同步36；debug多MOD阶段为37。hidden[14,16,3072]、context[14,512,3072]、FFN14336；候选私有可见性不变。投影及FA2中的QK/PV均使用BF16 Tensor Core MMA，softmax/RMS/RoPE不使用Tensor Core。

原ready退化、历次失败与数值限制仍保留；baseline、投机及原tile不动，四科学权限false。其他阶段融合和新调度尚未实施。
