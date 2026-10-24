# 官方机制依据

- NVIDIA CUDA GPU能力表：https://developer.nvidia.com/cuda/gpus —— RTX5090为CC12.0。
- NVIDIA Blackwell Tuning Guide：https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html —— SM120共享内存每SM128KiB、单CTA99KiB，区别SM100。
- CUDA13.0编程指南TMA：https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma —— tensor map、grid_constant、async proxy、mbarrier与128B swizzle。
- CUDA13.0 tensor map接口：https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html —— 本地实际cuTensorMapEncodeTiled、BF16与2D box约束。
- NVIDIA CUTLASS warp级MMA：https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html —— warp级mma.sync，SMEM通过ldmatrix进入寄存器；仅机制参考，不升级当前依赖。

本地实际编译CUDA13.0.88，模板CUTLASS3.5.1；不把SM100 tcgen05/TMEM套给SM120。TMA的实现/速度以本轮源码和完整block结果为准。无独立SHA任务。
