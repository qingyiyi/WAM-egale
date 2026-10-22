# GELU分块与FFN融合：源码核查

本次仅CPU核查/推导，未改计算代码、未启动GPU。此文的建议不是已经实现或实测的收益。原Torch是用户最新指定的后续唯一比较基线；历史原始证据保留。

## OBSERVED

- 第23项是单个Transformer block内的GELU阶段。源码`算子研究/wam_ops/optimized/csrc/whole_layer.cu:815`将FFN1的`[224,14336]`交给逐元素函数。
- 共3,211,264元素；`:633`固定每tile2048元素，因此1568tile。256线程每次向量搬4元素，每满tile各线程处理8元素。108个CTA静态分配，56个各15tile、52个各14tile，无每tile原子领取，也无每tilegrid同步；该阶段出口一次grid同步。
- `:708`附近GELU使用FP32乘加与`c10::cuda::compat::tanh`，输入/输出保持BF16边界。它不使用Tensor Core MMA；普通浮点/特殊函数的具体指令占比需当前SASS/NCU才能确定。
- FFN1为TensorOp SM80的128×256×32循环，8warp全部做MMA；FFN2由warp3–6做MMA，余线程协作写回。相关声明见`csrc/cutlass_gemm.cuh:15`及`:35`、`whole_layer.cu:195`及`:265`。
- 旧真实Torch GELU的原始profiler为`vectorized_elementwise_kernel<4,...GeluCUDAKernelImpl...>`，grid3136、block128，证据`pointwise-isolation-20261021/kernel-summary-gelu.json`。该历史布局不等于当前整层逐阶段重测结果；不能用CTA数多少直接推性能。

|每tile元素|逻辑tile数|每CTA最多处理tile|每满tile每线程元素|
|---|---:|---:|---:|
|2048（当前）|1568|15|8|
|4096（仅推导）|784|8|16|
|8192（仅推导）|392|4|32|
|16384（仅推导）|196|2|64|
|32768（仅推导）|98|1|128|

扩大tile不减少总元素/GELU次数、张量读写量或当前的那一次阶段grid同步，也不增加常驻warp。32768方案还会留下10个无工作CTA。循环和寻址可能变化，但不可保证加速。

## 融合的依赖与潜在收益

当前整层本来已在一个kernel内；新增优化应针对中间数据传递和阶段等待。

1. FFN1写回融合GELU：先按原次序加bias并舍入BF16，再转FP32算原GELU，最后BF16输出ACT。普通路径可省F1一次全局写和一次读，逻辑张量访问合计12.25MiB；这是访问字节推导，不是实测DRAM流量。GELU不再需要独立1568个工作包；FFN1→GELU屏障可移除，但FFN2读ACT前仍需合法就绪保障。debug所需F1可另保留。
2. FFN2最终写回融合残差3：必须在原两段K7168合并、bias及最终F2的BF16舍入完成后，按原公式计算gate与残差。不能在部分和上提前添加完整残差。可省普通F2中间读写和独立残差阶段，收益未测。
3. 四阶段tile流水：同一行块的一个FFN1输出tile只覆盖256个中间通道。完整FFN2归约14336通道，相当于56个生产tile；一个K7168分段覆盖28个。可以随依序就绪的K子块推进部分累加，无须等全部56个才开始，但必须保留原累加顺序与两段BF16边界。消费者多、跨CTA可见性、workspace生命周期及等待死锁均需实际设计。
4. 128行完整ACT的BF16数据为3.5MiB，不能整块常驻于A100单CTA shared memory；需分块流水或保留全局/L2中间布局。生产一个ACT tile可被多个FFN2输出列tile复用，不能为强行单CTA融合反复计算FFN1。

## HYPOTHESIS / UNKNOWN

不同就绪tile的Tensor Core MMA与普通浮点/特殊函数工作有重叠机会；单个warp不能越过自身数据依赖，固定warp分工不保证吞吐。当前FFN1八warp已都参与MMA，专门划出epilogue warp会减少现有MMA执行资源，需要测整体权衡。寄存器/shared/发射带宽也会限制重叠。

把GELU写在FFN1 epilogue可先减少中间读写和阶段开销，但只有另有就绪tile、双缓冲及合适warp映射时，才形成下一tile MMA与前一tile GELU的重叠；顺序调用不构成已实现的并行流水。

建议先做两个局部写回融合，再考虑跨GEMM流水；尚无新加速比。比较应包含FFN1+GELU、FFN2+残差3及整层相对原Torch，避免将融合后独立GELU阶段为0说成无限加速。

官方架构依据：[NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) 1.4.1.2–1.4.1.3（异步复制、生产者消费者与arrive/wait屏障）、1.4.1.5（BF16 Tensor Core）、1.4.1.1（shared/register资源约束）。这些硬件能力不证明当前实现已经形成流水。
