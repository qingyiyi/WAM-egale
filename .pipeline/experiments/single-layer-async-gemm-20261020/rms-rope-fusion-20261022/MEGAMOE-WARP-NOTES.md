# Warp角色与执行单元：本轮源码核验

这是技术解释，不是新的调度实现或GPU收益。参考固定DeepGEMM commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`，沿既有[原件及身份](../../transformer-tile-handoff-20261019/MEGAMOE-REFERENCE.json)；2026-10-22再次核对官方固定页面。

- BF16 MegaMoE的SM100路径：activation和weight TMA加载warp、MMA issue warp、scheduler、epilogue独立循环。MMA issue通过UMMA触发矩阵引擎，epilogue读取TMEM，做SwiGLU/转换/权重处理及写回等工作，不只是一次加bias。
- `kNumEpilogueStages=2`；MMA等待对应TMEM empty槽，计算完成发布full；epilogue等待full、读完归还empty。由此可以重叠不同tile的计算与后处理，而同tile消费者仍必须等待数据。启动/排空、消费者较慢或无就绪输入都会等待。固定角色不意味着全程无空闲。
- 非epilogue与epilogue角色显式采用不同寄存器分配。专职角色及有界缓冲用于流水和资源配置，不能只抄warp编号；本项目A100不具备这条SM100的UMMA/TMEM/TMA机制。
- 当前WAM CUTLASS路径以`OpClassTensorOp/Sm80`和MMA核心生成矩阵指令；普通表达式、FP64旋转、shuffle、load/store使用各自SIMT算术/特殊函数/通信/访存路径。`float4`和vec4本身不选择Tensor Core。判断执行单元应看PTX/SASS以及profile，不能看变量名或warp编号。
- 当前投影MMA warp3–6，所有8warp参与写回；shared mainloop/result使用union复用空间，且CTA阶段同步，尚无跨tile独立epilogue缓冲。历史warp7单独写回的版本不代表当前版本。

官方引用：
- https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh
- https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

本轮只做RMS+RoPE，不同时修改warp流水或GEMM。是否节省整层时间以本轮有限对照为准。
