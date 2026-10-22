# 当前单层Transformer：阶段同步与按需调制

当前optimized是旧11阶段同步版加一项按需调制融合，活动DSO41af9a65；一层一个persistent计算kernel，108CTA/256线程。没有跨阶段ready队列。

[完整阶段公式、tile数与同步](STAGE-MAP.md)是阅读入口。Python whole_layer.py绑定原27组权重；csrc/whole_layer.cu中的whole_layer按阶段执行，modulation4在消费处计算temb+固定表，point完成调制/残差。普通路径不分配完整MOD；debug额外生成表用于外部核验，所有正式单层计时用debug=false。

hidden=[14,16,3072]、context=[14,512,3072]、FFN14336。B14是14个独立Q16实例；不把候选变成可互相注意的Q112。保留共享prefix及private KV边界、RMS/Welford/RoPE/FFN分段BF16舍入。当前工作量uncached context，不以缓存本身当新贡献。

本轮完整层短/中/长2.269/2.435/2.449ms，较同轮旧11下降1.01%/0.98%/0.94%；同轮guarded约2.071/2.187/2.201ms，仍慢9.6%–11.3%。9真实签名和21结构有限逐位/状态通过，不能外推所有输入或机器人质量。[本轮证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)。

普通路径逻辑tile11912/11984、grid.sync38；debug保留MOD时13928/14000、39次。小投影M64×N128，context M128×N128，FFN1 M128×N256、FFN2两K7168段；其余分块没有改变。每CTA warp职责仍按阶段不同，详见STAGE-MAP和源码注释。

[上一轮ready调度负结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)有限精度通过但慢56%–64%，已回退，源码只在证据ready-source保留；不重新恢复。下一步按阶段讨论，暂未做Norm+调制等其他融合。baseline、投机生产和原tile不动，四科学权限false。
