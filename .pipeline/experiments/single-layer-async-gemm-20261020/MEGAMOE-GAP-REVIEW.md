# 当前B14单层实现与MegaMoE思想的差距

2026-10-20。按用户要求读取当前optimized源码与已有实测，对照固定DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`；本次未启动新算子GPU实验、未修改baseline/optimized。用户已明确原始单状态action保留原算子，以下优先讨论B14 verifier。

## 结论与已有工作

结构骨架已经实现：完整单层的一次cooperative kernel、108个常驻CTA、warp职责分工、设备内self/cross FA2、双缓冲cp.async GEMM，以及FFN行组就绪交接。有限数值对齐也已完成。尚未实现MegaMoE那种细到K片段的数据就绪流水、独立运行的角色循环与有界中间槽回收，也没有实测新增性能收益。不能给“完成百分比”或把结构相似当效率相同。

当前默认B14短/中/长KV为12.9445/10.8687/11.2424ms，native为2.5778/2.4073/2.8418ms；耗时分别5.022/4.515/3.956倍。ready还增加约2.5%–2.7%。本轮是uncached eager完整单层工程计时，每格4个样本；不是强cached/Graph、30层verifier或E2E。[最终原件](FINAL-LAYER-RESULT.md)

## 逐项映射

|机制|当前WAM单层|固定MegaMoE实现|剩余工作|
|---|---|---|---|
|常驻执行|一个kernel、108CTA，阶段内领取tile|持久角色循环及任务metadata流水|已有结构，不能仅继续减少launch|
|warp分工|0调度、1载入A、2载入W、3–6计算、7写回；FA2由计算warp自行load/compute/store|activation/weight/Tensor Core issue/scheduler/epilogue独立循环|当前角色频繁在同一CTA barrier会合，生产/消费解耦不足|
|GEMM内流水|双缓冲cp.async，M≥128为64×128×64|多阶段full/empty barrier与异步TMA/UMMA|SM80须以自身MMA/cp.async实现，不能直接复制SM100指令|
|跨算子依赖|FFN须完整64行组的14336列全部ready才开始FFN2|L2按当前K片段等待L1对应完成位|在保留原归约与舍入下实现K片段ready，避免所有CTA等待未调度producer|
|workspace生命周期|多数中间量完整落global，完整调用结束回收|有限live pool、ring generation、生产/消费归还|证明最小槽数与释放条件；不声称MegaMoE把全部中间量留在shared|
|资源分配|全kernel255regs/thread、121376B shared、理论1CTA/SM|SM100角色寄存器重新分配、TMEM和双CTA机制|A100不能靠同样指令实现；需编译期专用化与缩短活跃变量生命周期|
|候选相关复用|linear已有M224；attention仍14个独立Q16，合法历史地址复用|专家/通信特定调度|WAM尚无CTA内跨候选共享query tile；private KV必须隔离|

外部源码：[K片段依赖与scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L143)、[warp角色与流水](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh#L626)。旧源清单在[MEGAMOE-REFERENCE](../transformer-tile-handoff-20261019/MEGAMOE-REFERENCE.json)。

## 为什么慢：证据分层

**已测量。** 完整单层确实慢3.96–5.02倍，ready负收益已在两个独立进程观察。旧M38原型阶段profile显示GEMM主导，B14 cross-K/V很重；但旧原型尚未使用当前FA2/双缓冲，旧66%–85% GEMM占比不能冒充当前版本占比。新版本没有NCU stall/带宽/实测occupancy数据，不能给单一慢因的百分比。

**当前源码能确定的成本。**

1. `whole_layer.cu:69–96,262–302`仍按约29个数学阶段推进，阶段开始/结束有grid同步。大多数中间结果写完整global workspace，融合主要消除了host launch，尚未普遍消除中间读写与阶段等待。
2. `gemm_pipeline.cuh:57–106`虽已双缓冲，每个K块仍有CTA同步；结尾计算warp将FP32 accumulator写shared，再由单个warp逐元素做bias/BF16与global写回。B14宽tile有8192元素，32线程写回，未与下一tile计算形成独立epilogue流水。该事实不等于已经测出写回占比。
3. `whole_layer.cu:45`的宽GEMM存储含`partial_y[64×128]`，占32768B；它在`gemm_pipeline.cuh:50,91,100`只服务于`m==32`的交错归约，而宽路径只在`m>=128`调用。这是明确的B14专用化机会。当前宽GEMM共享结构120832B；去掉这32KiB仅使该结构到88064B，仍不自动满足双CTA驻留，不能承诺occupancy翻倍。是否降低寄存器与实际延迟需编译/测量。
4. B14 action projection实际M224，64×128输出tile共4×24=96个，少于108个常驻CTA；M方向填充利用率224/256=87.5%。这是几何量，不能当实测SM利用率。简单再放大tile可能更少任务、更高寄存器占用。
5. `whole_layer.cu:285–288`每调用重做cross K/V：M=14×512=7168，单个投影2688tile。按2MNK算术量，两次cross K/V为270.583GFLOP，占这条uncached路径全部Linear理论FLOPs约80.67%；不是耗时/HBM占比。实际强基线已经可使用条件缓存，必须给两边同等缓存条件，再讨论新调度增量。
6. 当前FA2已经在self/cross内部，不能再用旧朴素softmax的慢因描述新版。它仍嵌在统一高资源kernel中；B14无跨候选CTA内共享历史读取，也没有证明实际HBM流量减少。

数值修复要求保留cuBLASLt观察到的BF16分段/归约与bias顺序。它约束可选流水，但尚无消融证明“逐位一致要求”本身导致4–5倍退化；不能以放宽旧门代替实现优化。

## 建议执行顺序

1. **先固定B14的强基线和当前慢点。** 同真实短/中/长输入、相同cross-KV缓存/Graph条件，记录完整block及GEMM/attention/pointwise/等待。原始B2 action继续原算子。补的是当前版本归因，不重用旧M38比例。
2. **B14专用化和基本tile效率。** 消掉宽路径不用的小M缓冲/分支，检查第二组accumulator是否仍产生资源负担，优化共享布局、向量写回与有限流水深度。先将真实3072/14336归约形状的GEMM接近成熟库，再接完整层；保持BF16契约。不要先扩大任务队列。
3. **再推进K片段ready。** 让FFN2按固定K顺序消费已经完成的FFN1/GELU片段，给生产者保留推进机会；槽generation、release/acquire、归还协议齐全。WAM GELU不减半，不能照抄MegaMoE SwiGLU的BLOCK_N/2公式。
4. **按Transformer依赖扩展。** Q/K/V与attention按head/合法candidate推进；attention输出投影、Norm依赖完整所需维度，不能逐tile随意跨越。self和cross都纳入DAG，不能只完成FFN就声称整层数据流已完成。需要复用历史时先证明同CFG/条件可共享，保持私有suffix。

前三步是否带来增量由完整block与强baseline决定。B14基本原语仍落后数倍时，扩大ready队列或增加persistent角色不是优先项。若最终没有稳定收益，保留单kernel/数值结果但收缩系统性能主张。

代码入口均在`/rjs/cxz/论文探索/算子研究/wam_ops/optimized/`：`csrc/whole_layer.cu`、`csrc/gemm_pipeline.cuh`、`csrc/ffn_pipeline.cuh`、`csrc/fa2_stage.cuh`。本报告为审查与建议，不宣称这些后继已实现或新实验已启动。
