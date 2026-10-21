# Torch投影CTA/warp与常驻调度核查

本次只读现有原版trace、10/21 NCU、两组8warp结果及固定MegaMoE源码；未新增GPU作业，M39累计仍2073.412951秒。

## OBSERVED：原版真实配置

依据10/19三KV原版Torch profile（B2/B14各3份），GEMM顺序为self Q/K/V/out、cross Q/K/V/out、FFN1/FFN2。三KV每类grid/block均相同。10/21 B14中KV NCU独立核对Cross K、FFN1/FFN2一致。原始文件路径及源身份见TORCH-PROJECTION-LAUNCHES.json。

|阶段（每次投影）|B14 grid|B14 CTA数|B14 warp/CTA|B2 CTA数|B2 warp/CTA|
|---|---|---:|---:|---:|---:|
|self Q/K/V/out及cross Q/out，各自|24×2×2|96|4|48|4|
|cross K、cross V，各自|24×28×1|672|8|96|8|
|FFN1|56×2×1|112|8|112|4|
|FFN2|24×2×2|96|4|84|4|

warp数是block线程数/32；不能仅凭launch配置证明每个warp逐条执行MMA或还原其内部职责。B14 context模型矩阵M7168/N3072/K3072，原版kernel名字128x256，按模型行列口径推导输出覆盖M256×N128；cuBLAS kernel名称的维度方向不等于模型行列名。小action为M224，不共用这一tile。

## 672个CTA与108个常驻CTA的区别

672是一个kernel launch的总CTA数，不是672次kernel launch，也不是同时672个CTA驻留。原版Cross K的256线程、238regs/线程、约145KiB shared，使本次108SM A100每SM至多一个这类CTA；约6.22个满波工作量，由硬件滚动补位，不按整轮全局同步。108常驻CTA也可各处理约6或7个tile；计算量足够时，CTA总数本身不构成速度结论。

|执行|总启动CTA|context逻辑tile任务|
|---|---:|---:|
|原版Cross K|672|按shape/grid对应672个256×128输出块|
|选中自有4warp|108常驻|1344个128×128块|
|已否决自有大tile8warp|108常驻|672个256×128块|

原版硬件在CTA退出后调入下一个，CTA间不需软件atomic队列或persistent阶段同步。常驻实现省的是跨算子kernel launch、可能的数据重用，而不是消除原版672次launch（原本就只有一次）。当前候选同时改变tile、warp映射、编译资源/函数状态与任务组织；没有同一计算体仅改CTA数的对照，因此不能断言672>108，更不能把原版全部优势归于硬件调度。

## NCU支持什么

B14中KV旧4warp、独立hot Cross K：

|指标|native|自有|
|---|---:|---:|
|Tensor管线活跃|88.50%|70.29%|
|活跃warp比例|12.49%|12.50%|
|寄存器/线程|238|255|
|DRAM bytes|207410304|350640000|
|local路径读写|0|28753920 bytes（27.421875MiB）|
|barrier warp-stall|6.26%|46.64%|

这些支持优先检查MMA流水、访问复用和设备调用状态。NCU barrier是warp级停顿比例，不是整层墙钟等待占比；local包含已观察到的函数保存/恢复，不能全部叫allocator spill，也不能当成HBM流量。两者活跃warp比例近同，不支持仅用occupancy解释性能差。

8warp两候选尚无NCU计数，旧4warp指标不能解释它们各自退化的确切原因。大tile增加shared且local静态字节略变；固定tile共享资源不增仍慢1.05%–1.33%。warp tile变小可能改变共享读取和指令/数据复用，属HYPOTHESIS，不能提升为已观察根因。

## MegaMoE与当前设计建议

核查DeepGEMM固定1fc9c66975ee322d628b3288534ca0e9ce12a64d：SM100 BF16实现626/687/726/816/826行，activation TMA、weight TMA、MMA issue、scheduler、epilogue分别有独立任务循环和full/empty交接，角色固定而领取的Linear1/2 tile动态变化。专职线程让不同硬件管线重叠；源码还有不同角色的寄存器重分配。以上是源码结构事实；“减少切换状态/方便局部流水”是设计解释。SM100 UMMA/TMA/TMEM与A100 SM80不同，不能照搬单issue warp或warpgroup寄存器重分配。

建议：按算子类型选择预编译的warp分工，处理一个tile及其在途buffer时角色固定；tile安全结束后可切换任务类型。GEMM/FA2/RMS/逐元素无需强制一套0/1/2/3–6/7分工。进入有真实重叠的pipeline后，其producer/consumer身份保持稳定。此为讨论建议，未作为用户已选择的设计写入长期约束。

## 可证伪复现步骤（建议，未运行）

1. 同真实context输入、BF16边界和tile，先独立device GEMM匹配原版覆盖/warp/流水及epilogue，保持同前序缓存条件。
2. 尽量同一编译计算体：每tile一个CTA（672）对108CTA静态grid-stride；再108CTA原子队列。记录实际任务遍历/缓存差，拆硬件分派、循环、atomic与设备状态成本。
3. 对齐独立原语后再嵌入整层；独立快、嵌入慢才继续查统一寄存器/shared配置、设备调用活跃状态与同步。不要一开始用跨阶段调度掩盖原语差距。

可复现映射与调度结构；同grid/block不保证同机器指令、访存复用或同速度。Torch CUDABlas.cpp主要调用cuBLAS/cuBLASLt，不提供被调用厂商GEMM设备体，复制Torch封装不能把它变成现有persistent kernel内的可调用tile。

参考：
- 本地NCU：../remaining-cost-20261021/local-ncu-exclusive/NCU-SUMMARY.json
- 本轮投影对照：RESULT.md；原版launch摘要：TORCH-PROJECTION-LAUNCHES.json
- MegaMoE固定源码：https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh
- Torch2.9调用源码：https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CUDABlas.cpp
