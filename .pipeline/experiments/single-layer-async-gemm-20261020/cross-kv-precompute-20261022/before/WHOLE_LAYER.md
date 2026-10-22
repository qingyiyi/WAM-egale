# 当前单层Transformer：Cross K/V分组复用

一个persistent完整层kernel，保持阶段全grid同步；已做按需调制、self RMS+RoPE融合、新增按相同context分组复用Cross K/V。活动DSO `cc2656e76444a7b9f32ec29acdbf57ba94d9c5d0120f1961eb9f7a5d130e1593`。没有恢复ready调度。

先在Graph外调用`WholeLayerBlock.prepare_context(context)`，本批B14得到两组条件；改变context后必须重新prepare。self历史行映射独立。普通调用每次计算两组K/V及K RMS，Cross attention的14个Q16分别读取所属组K/V。没有跨调用缓存。无prepare时保持完整形状路径；B2兼容。

有限9签名/27结构逐位与有效KV/RNG通过。两进程同轮完整层短/中/长2.258/2.424/2.442ms→1.205/1.372/1.388ms，降低46.63%/43.38%/43.17%。同样分组重算的guarded为1.226/1.345/1.360ms，warm cached guarded为0.958/1.078/1.091ms，不能宣称超过最强缓存baseline。

[阶段表](STAGE-MAP.md)、[证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-reuse-20261022/RESULT.md)。G2/G1为6200/6272逻辑tile，36个grid同步；资源108CTA/256线程/100096B shared/255regs/static local936B。投影和attention QK/PV使用Tensor Core；归约/逐元素使用CUDA core。RMS没有融合进投影。

基线、投机、原tile不动；四科学权限false，有限单层验证不等于完整模型或闭环质量。下一建议为两组最终Cross K/V预计算，双方同等缓存、冷warm分列，尚未实现。
