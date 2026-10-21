# 官方实现核验与本轮取舍

本轮只借鉴SM80可用的访存、归约与线程映射；保持目标模型原公式、归约树、BF16边界和RoPE FP64。

|来源|已核验机制|本项目采用或限制|
|---|---|---|
|[PyTorch固定版本RMS](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/native/cuda/layer_norm_kernel.cu)|vec_size4、每行128线程、warp及shared归约、FP32累计|直接复用本项目已固定的设备数学体；裸RMS已与Torch接近，优先去除额外shared中转|
|[PyTorch逐元素](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/native/cuda/CUDALoops.cuh)|按对齐选择向量化访存，多个元素由每个线程处理|调制、残差按4元素打包；显式保留mul/add顺序，避免融合FMA改变舍入|
|[FlashInfer RMS](https://github.com/flashinfer-ai/flashinfer/blob/07f921aeba912e6ebfabe2c04b509a591b6152b7/include/flashinfer/norm.cuh)|对齐向量加载、平方和归约、小shared归约空间；含fused-add变体|其xor归约和向量宽度与固定Torch树不同，不能直接宣称逐位等价；本轮不导入整个库|
|[FlashInfer位置编码](https://github.com/flashinfer-ai/flashinfer/blob/07f921aeba912e6ebfabe2c04b509a591b6152b7/include/flashinfer/pos_enc.cuh)|位置编码的向量数据组织和专用kernel|模型采用预计算complex64频率和FP64复数乘法；不套用不同sin/cos或FP32算术|
|[TransformerEngine RoPE](https://github.com/NVIDIA/TransformerEngine/blob/5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a/transformer_engine/common/fused_rope/fused_rope.cu)|融合RoPE读写、按布局/stride映射；可处理交错旋转|其常用路径float/sincosf和本项目契约不同，只借鉴融合数据组织；保留原FP64|
|[CUTLASS3.5.1 epilogue](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cutlass/epilogue/thread/linear_combination_generic.h)|片段级线性组合和向量化输出|适合后续GEMM后处理；不能跳过原GEMM输出BF16再门控的边界，本轮先独立小算子|
|[DeepGEMM/MegaMoE](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)|单mega-kernel融合/重叠通信与多阶段计算；当前官方需求SM90/SM100|本机A100为SM80，只作为就绪任务/工作分解参考；不是RMS/RoPE设备体的直接替换源|

cuBLASLt官方文档提供预定义bias/activation epilogue：[CUDA12.6官方文档](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublasltepilogue-t)。它不是通用RMS/RoPE设备函数源码；这类小算子优先核验Torch/FlashInfer/TransformerEngine。

## 用户确认的工作分解

每类算子按合适tile大小和逻辑任务数执行；108是当前resident worker池规模，不是每阶段必须凑满的任务数。后续由真实DAG就绪条件，把无依赖或依赖已满足的后续tile提前分配给空闲worker。当前小算子核对及顺序版测量不冒称已实现跨阶段ready。

具体commit、文件SHA256、下载URL与失败的API探针保存在REFERENCE-SOURCES.json。API限流后使用git ls-remote取得实际HEAD，再下载固定SHA；不把main标题当冻结源。
