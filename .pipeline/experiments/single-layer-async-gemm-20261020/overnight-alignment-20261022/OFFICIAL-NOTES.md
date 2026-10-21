# 原语对齐的官方边界

2026-10-22重新核查[NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) §1.4.1.1、§1.4.2.3：SM80每SM有64K个32位寄存器，每线程最多255；shared/L1共用192KiB，支持100/132/164KiB等shared划分。当前每线程255、256线程以及约100KiB shared，实测只可驻留1 CTA。较大shared会限制可用L1，但具体运行时划分与退化贡献仍需counter，不据理论容量冒充实测。

[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)说明tile大小、并行度和资源需权衡；epilogue常经shared交换来实现合并写回。因此不能只因有shared中转就认定实现错误，也不能认为直接寄存器scatter一定更快。代码继续绑定本地CUTLASS v3.5.1，不随文档latest升级。

本轮所有收益与失败均以真实输入计时/逐位核验为准。上述材料只提供机制解释边界，不提供本项目加速或误差保证。

## SM80不支持动态setmaxnreg

2026-10-22核对NVIDIA官方PTX ISA8.3归档：setmaxnreg要求sm_90a，不能用于本项目A100/SM80。它不是可直接照搬到当前persistent warp的寄存器动态重分配办法；不能因此推断任何特定新GPU实现一定更快。本轮没有迁硬件或改驱动。源：https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg （对应官方PDF：https://docs.nvidia.com/cuda/archive/12.3.2/pdf/ptx_isa_8.3.pdf）。本轮本地编译仍CUDA12.4/CUTLASS3.5.1。

## 参数传递核查

最终源码已使用`const __grid_constant__ Params p`，也已避免为输入残差复制Params；不是本轮缺失项，不追加相同优化。NVIDIA CUDA12.4指南说明该标注阻止按引用使用kernel参数时生成线程私有副本。最终SASS仍有LDL/STL指令位点，但静态位点不能代替动态执行/流量归因。来源：https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_C_Programming_Guide.pdf；本地核查PARAMETER-COPY-CHECK.json。
