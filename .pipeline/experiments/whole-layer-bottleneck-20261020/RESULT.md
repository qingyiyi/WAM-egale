# M38：整层单kernel慢点实测与源码解释

结论：当前自有整层原型慢在设备计算组织，首要是GEMM，其次是长历史attention。减少launch已经实现，但不足以补偿成熟cuBLAS/FlashAttention被较朴素WMMA与同步流水替代的代价。本轮没有改主实现，没有解决原逐位失败。

OBSERVED：远端A100-SXM4-40GB GPU0，2 CPU核，同卡无观察到其他context；6真实action签名，原Torch/native逐位6/6通过，whole原逐位6/6失败；profile/static副本全部中间量、输出、有效只读KV、tile exactly-once、输入/RNG匹配失败whole。仅只读fixture，不扩大到合法写、30层或机器人。

## 同轮完整block计时

以下是独立于profiler的CUDA event样本，每格一进程六轮交错中位数。计时输入与本轮数值验证是同一个fixture对象，每次恢复原缓存。native保持成熟Torch/cuBLAS/FlashAttention计算链并在本轮核对原Torch输出；不是直接Python Torch的wall-clock列。双方均未缓存cross K/V，未使用最强cached/Graph；不能作为新算子准入性能或论文加速。

| 真实输入 | native ms | whole ms | whole/native |
|---|---:|---:|---:|
|short-B2|1.481|7.532|5.09×|
|middle-B2|1.630|8.638|5.30×|
|long-B2|1.755|9.774|5.57×|
|short-B14|2.230|24.830|11.14×|
|middle-B14|2.501|26.934|10.77×|
|long-B14|2.932|30.684|10.46×|

原whole实际只有1个计算kernel；native B2有39个、B14有35个计算kernel，另外有memcpy/memset。原始worker字段kernels来自所有CUDA events，含memory事件；最终统计由Chrome trace cat=kernel重新分类，不能把41/43个设备事件全部叫计算kernel。component::注释也不是kernel；本报告通过External id关联CPU作用域，只统计真正GEMM kernel，保留内存事件和注释范围原值。

## 阶段瓶颈

globaltimer插桩副本保持原whole的数值/状态，但编译寄存器172→175且实测速度改变。下面比例是该副本两次audit-off设备采样均值（第三次audit-on仅保存，不混入）；阶段含边界等待。不得直接从原whole event延迟减去这张表，或声称指令级归因。

| 输入 | GEMM | 其中cross K/V | self attention | cross attention | 其余/等待 |
|---|---:|---:|---:|---:|---:|
|short-B2|80.1%|40.5%|4.7%|3.4%|11.8%|
|middle-B2|69.9%|35.5%|15.6%|3.0%|11.5%|
|long-B2|60.8%|30.8%|25.4%|2.6%|11.2%|
|short-B14|84.5%|63.3%|3.6%|3.6%|8.4%|
|middle-B14|76.3%|57.2%|12.8%|3.2%|7.7%|
|long-B14|66.1%|49.6%|24.5%|2.8%|6.6%|

同一个whole真实中间张量送入原Torch Linear，下面比较整阶段与原Torch GEMM kernel。前者含领取/barrier；后者单kernel，内存初始化另列，不是纯MMA指令等价计时。完整60行见[COMPONENTS.csv](COMPONENTS.csv)。

| 输入/组件 | M,N,K | whole阶段 ms | Torch GEMM ms | 阶段/kernel |
|---|---|---:|---:|---:|
|short-B2/Q_proj|32,3072,3072|0.2473|0.0241|10.25×|
|short-B2/cross_K_proj|1024,3072,3072|1.3481|0.1058|12.74×|
|short-B2/FFN2|32,3072,14336|0.7665|0.1161|6.60×|
|short-B14/Q_proj|224,3072,3072|0.4609|0.0420|10.97×|
|short-B14/cross_K_proj|7168,3072,3072|8.1356|0.7138|11.40×|
|short-B14/cross_V_proj|7168,3072,3072|8.5307|0.7135|11.96×|
|short-B14/FFN1|224,14336,3072|1.5454|0.2065|7.49×|
|short-B14/FFN2|224,3072,14336|1.5183|0.1305|11.63×|

GEMM合计：B2原型约5.41ms，而相同输入原Torch GEMM kernel总和约0.575–0.577ms；B14约22.25–22.36ms，对应原Torch约1.590–2.019ms。它们来自不同profile调用，解释量级而非严格相减归因。B14各完整计时样本有约20%级波动，原始144条全保留，未锁时钟，不把小百分比当正式收益。

长B14 self-attention：原型阶段8.301ms，对照native内FlashAttention kernel0.328ms；长B2为2.256ms，对照split-K forward+combine约0.063ms。后者没有计入native单独pack等准备，因此也不是完整attention接口的纯公平倍数；完整block表已经包含全部工作。

## 调度反事实

唯一改动：每stage的atomicAdd领取改成CTA grid-stride。两者都有相同globaltimer插桩；均175寄存器、59616B shared、256线程、108CTA、理论1 CTA/SM。所有有限输出/状态一致；无padding/grid/seed搜索。

| 输入 | 六个同repeat静态/动态比的中位数 |
|---|---:|
|short-B2|1.0397|
|middle-B2|1.0317|
|long-B2|1.0312|
|short-B14|1.1216|
|middle-B14|1.1280|
|long-B14|1.1033|

六种签名配对中位数均变慢3.1%–12.8%。长B14有一对静态更快，全部原样保留；不能只看某个最优轮。这个实验不支持“去掉全局队列就能解决5–11倍差距”。它没有删除grid/CTA barrier，也不证明全局atomic指令成本为零；静态分配的分布/就绪与编译细节也会改变。

## 从源码能确认的原因与尚不能确认的部分

1. [gemm_tile](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu#L78)为64×64×32、单shared buffer。warp1/2先同步global→shared，CTA barrier后warp3–6做MMA，再CTA barrier；没有加载下一K块与当前MMA重叠。K3072循环96次，FFN2 K14336循环448次，单tile内部至少194/898次CTA barrier（另有领取/完成同步）。这是代码事实；各类barrier具体stall占比UNKNOWN。
2. B2的M32仍用M64 tile，逻辑行利用率50%；B14 M224需256行容量，87.5%。B2 D×D投影只有48个输出tile，不能同时让108个常驻CTA都有有效tile。不能把这些静态比例当实际SM利用率。
3. B14 cross K/V实际M=14×512=7168，超过action M224。原型每次重算；短B14这两项占插桩kernel63.3%。已有prompt/cross-KV缓存是强baseline的一部分，省掉它不能叫新MegaTransformer贡献。当前没有实现跨候选attention的共享query tile，也没有共享private KV。
4. [attention_tile](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu#L264)每CTA只处理一个candidate/head的16Q、每次64KV。QK/softmax/PV/在线输出之间有多次全CTA同步；softmax只有16个线程，每线程串行扫64元素；KV循环没有当前MMA/下块加载重叠，也没有本版KV split并行。长历史成本放大。
5. [whole_layer](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu#L328)29阶段依次推进，每阶段全grid同步，中间量落global workspace。因此现有实现有单kernel/warp分工，尚无跨阶段细tile生产者消费者流水。
6. 原kernel172 registers/thread、59616B shared，仅允许1 CTA/SM，理论8/64warps=12.5%，不是实测occupancy。但原Torch的大cross-KV GEMM trace也报告238 registers/thread、147456B shared、256线程，只能一CTA驻留，却快得多。因此不能用“occupancy低”单独解释差距。不同原语的数据通路、流水和tile复用更值得先处理。

UNKNOWN：NCU此前无权限，本轮不提权重跑；L2/DRAM bytes、bank conflict、实际occupancy、指令stall分解均未测。Torch trace的est. achieved occupancy=0是工具字段，不能当实测0%。当前没有证明HBM流量下降、计算受限/带宽受限、队列唯一根因或系统正协同。

## 下一步建议

优先修单个GEMM tile的数据通路：成熟SM80布局、异步多缓冲加载、warp计算流水，同时保留必要BF16舍入边界；先用本次真实M/N/K清单做同输入原语与数值门，再接回整层。先测基本tile能否接近库的速度，再投入更复杂依赖队列。不能只把M37未通过逐位的CUTLASS组件塞回整层。

其次处理attention的warp级softmax归约、KV加载/计算流水和有依据的KV分块。然后在同等合法cross-KV缓存条件下比较整层，最后才研究跨阶段就绪队列与中间量驻留。目标仍是一个单层kernel，不改成30层；baseline冻结。这个顺序是基于本轮profile的建议，未自动启动下一轮实现。

本轮[源码讲解](IMPLEMENTATION-GUIDE.md)、[原始结果](remote-evidence/profile-1/RESULT.json)、[trace精确分类](TRACE-ATTRIBUTION.json)、[完整计时](TIMINGS.csv)、[资源](RESOURCE-AND-STOP-REPORT.md)。全部历史质量门false，原数值FAIL不改。
