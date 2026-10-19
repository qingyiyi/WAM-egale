# 短KV/B14：GPU复测前已核实的范围

## 已观察

1. 旧M22的短KV/B14 action六个self Graph-core点，新/Torch为1.0089–1.0117；cross为1.0036–1.0102。旧完整block六点为0.9542–1.0838，其中两点慢超过5%。34.37%最大回归实际是long/B2/layer8，并非short/B14。旧原始值全部保留，见OLD-TIMING-GROUPS.json和OLD-SLOW-ROUNDS.json。
2. 较新的M25两进程short/B14完整block比为0.9997/0.9818，完整30层为0.9984/0.9735；没有在这两个新点复现旧6%–8%退化。仅block0/first六fixture覆盖，不能排除原layer29/last的慢点；NVML异常仍在，不能签正式性能。
3. 固定Torch 2.9源码与独立CUDA入口均使用SM80 D128非causal标准tile128×64、split tile64×128，且同一occupancy split选择公式。B14×H24下从公式导出split=1。输入要求contiguous且没有内部QKV拷贝，当前wrapper直接用原stride。编译器CUDA12.4/12.6差异保留；尚未证明SASS相同。
4. baseline完整block forward与原模型forward的AST相同；优化默认只替换SDPA，没有新增FFN数学或缓存优化。所谓完整block慢点不能仅凭结果归因于SDPA核函数。
5. 复用M26实际short/B14 profile，GPU kernel总和2970.389us，其中两个cross条件K/V GEMM各715.454/714.397us，合计48.14%；缓存这两个原shape原Linear输出后，strong路径kernel总和1185.148us。两次原始投影处理M7168，而action线性M224；这是固定512-token条件在7候选×CFG2上的重复工作，不是action horizon变长。
6. 相同guarded+cached路径旧eager中位2.353856ms、Graph1.212864ms；dense+cached为2.663264/1.531328ms。不同执行边界不得混为“自研算子对Torch退化”或新MegaTransformer贡献。profile时长不进入性能表。

## 尚不能确定

- 旧完整block慢点的具体根因：原始采样有明显双向/跨轮波动，但缺少同步的每个慢样本完整timeline；不能直接判定都是CPU争用、GPU频率或数学kernel。
- 独立CUDA约1%核心差异是否稳定、是否来自编译器、寄存器或计时分辨率，需要同一实际Q/K/V下Torch A/A与独立CUDA交错复测。
- 如果用户所指为“B14绝对延迟比B2大”或“融合收益不足”，其比较对象不同；已请求可选澄清，独立继续核查。

## 下一步

新tools/diagnose_attention.py预备两次独立进程，记录真实六fixture的self/cross核心、完整未缓存block及同等缓存/Graph block。Torch_A与Torch_B是完全相同操作，用于量化测量本身的波动；不按结果删慢样本。完整输出、有效KV、metadata、输入和RNG门保持逐位标准。当前foreign GPU占用，未启动GPU作业；不与对方抢计时，也不修改时钟。
