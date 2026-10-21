# 为什么写 bank conflict 降到 0

## OBSERVED：两项代码改动，指标含义分开

当前保留的改动来自上一轮，不是本轮又改数学实现：

1. `算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh:14` 将临时累加矩阵stride作为模板参数，context实例使用136；逻辑输出仍是128列。`:80` 用同一stride存入shared，`:93` 用同一stride读出，写回全局的逻辑坐标不变。`gemm_pipeline.cuh`通用context分派与`whole_layer.cu:176`显式context入口均使用相同实例。
2. `cutlass_gemm.cuh:85` context epilogue每线程搬4个float，减少读取/循环指令；FP32加法顺序与BF16舍入保留。SASS确认shared读为LDS.128；全局结果写仍为两条32-bit store，不能称64-bit全局写回。

shared-store bank conflict降到0由第一项padding已实现，不能将它归因于第二项向量化。此次核对没有将其他FFN共享布局一并改为136，未引入ready-DAG。

## OBSERVED：地址枚举与原NCU完全相符

A100共享内存有32个bank，连续32-bit字分布到连续bank。对本次float数组，`bank=(row*stride+col)%32`。硬件原则参见[NVIDIA CUDA Programming Guide 13.2.1](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/02-basics/writing-cuda-kernels.html)和[Best Practices](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-c-best-practices-guide/index.html)。

本地固定CUTLASS v3.5.1 `mma_tensor_op_tile_iterator.h:3293`的累加器映射为`row=lane/4`、`col=2*(lane%4)`，每线程本条STS.64写2个float。以一条指令中前16个lane负责的128B子事务说明：

|线程|相对行|列|stride128的bank|stride136的bank|
|---|---:|---|---|---|
|0–3|0|0–7|0–7|0–7|
|4–7|1|0–7|0–7|8–15|
|8–11|2|0–7|0–7|16–23|
|12–15|3|0–7|0–7|24–31|

原stride128对32取余为0，四行的不同地址竞争同8个bank，形成4路冲突。136对32取余为8，四行错开成32个不同bank，每个bank一个字。另半warp有自己的理想128B事务；跨这两个事务出现相同bank不属于额外冲突。这里只说明本次指令和映射，不能把所有warp访问一律当16lane执行。

`bank_address_audit.py`沿实际warp tile偏移和全部64条STS.64枚举地址；[机器可读结果](BANK-ADDRESS-AUDIT.json)核对了原NCU的K/V两核：

- 理想写wavefront：672 CTA × 8 warp × 64条 × 2 = **688128**。
- 原4路冲突写wavefront：688128 × 4 = **2752512**。
- 额外冲突：2752512 − 688128 = **2064384**。
- padding后实测写wavefront688128、额外写冲突**0**，K/V均一致。

这是源码地址分布、SASS与硬件计数三者相符的局部机制证据。0仅指这两个真实独立投影kernel的shared写冲突；shared读仍有少量冲突，计算/访存/同步耗时仍存在。当前4warp整层没有新增NCU证据，不能宣称整层所有shared冲突为0。

## 验证与性能范围

逻辑矩阵、MMA和BF16舍入不变；padding与更宽搬运属于布局/指令改动。上一轮九个真实签名的完整层输出max_abs0、共有中间量/有效KV/输入/RNG有限门通过，不能外推所有输入。共享存储是union，扩大的结果区仍小于MMA主循环区，所测shared资源上限未增加。

独立8warp投影约615→593–594→577µs；当前4warp整层短/中/长KV约2.660→2.560、2.872→2.774、3.124→3.025ms。完整层耗时下降3.12%–3.77%，不是完整任务速度或所有算子对齐。原始结果见[上一轮RESULT](../projection-gemm-core-20261021/RESULT.md)，失败与无效对照保留。
