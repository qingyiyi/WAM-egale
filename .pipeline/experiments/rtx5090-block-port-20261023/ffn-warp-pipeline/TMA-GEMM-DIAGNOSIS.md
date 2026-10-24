# FFN与输出投影：TMA负结果的源码诊断

2026-10-23晚至10-24 00:04，CPU源码/既有结果分析；没有新GPU实验、编译或性能数据。当前保留版本仍为GRAIN-M64-FFN2，三个TMA候选仍不采用；原预算、终态、四项科学权限和production_admitted=false不变。

## 观察到的慢点

下表复用22:25前的GRAIN阶段诊断，非本次重测。比值为原Torch对应区域/自有嵌入阶段（含阶段同步），不是独立纯GEMM吞吐。完整17阶段见[原阶段表](STAGE-TABLE.md)。

|阶段|短/中/长KV比值|自有相对Torch延迟|
|---|---|---|
|FFN1+GELU+FFN2+残差3|0.750/0.748/0.749|约高33%–34%|
|Self输出投影|0.682/0.677/0.680|约高47%–48%|
|Cross Q投影|0.699/0.699/0.696|约高43%–44%|
|Cross输出投影|0.686/0.683/0.684|约高46%|

中KV原始诊断：FFN为自有0.341280ms/Torch0.255178ms；Self-out为0.061696/0.041790ms；Cross-Q为0.059584/0.041648ms；Cross-out为0.061136/0.041734ms。FFN占软件记录阶段跨度总和约39.54%，另三个投影合计约21.13%；这不是独立无插桩整层的可加性能模型。数据源：`../blackwell-tma/reports/GRAIN/SUMMARY.json`。

最后同场无插桩完整层原Torch/GRAIN仍为1.0892/1.5417/1.9568倍；整层收益不证明FFN局部已经对齐。INPUT/EPILOGUE/KREADY同场均退化，见原REVIEW_RETURN与各自SUMMARY；不能把跨轮微小差异当作角色交接的单因素效应。

## 已核实的实现差异及其边界

1. **TMA原型没有完整保留原CUTLASS主循环。** 当前GRAIN的FFN2使用三级shared输入流水；生成的`../build/generated-normal/generated/cutlass_gemm_mainloop.cuh:506`起在当前MMA前加载下一组寄存器fragment，并交错发cp.async。`epilogue_pipeline.cuh:145`起却是当前fragment加载→transform→MMA，源码中没有同样的下一组寄存器预取；输入变为两级。这些是可见代码差异，编译器最终重排及各项耗时占比尚未测量，不能断言TMA指令自身更慢。

2. **输入异步不等于计算链都被隐藏。** TMA解决global→shared；现行BF16 `mma.sync`仍读寄存器，shared→register预取也需要流水。当前每FFN2输出tile有224个K64轮次，每轮2次TMA请求、2次128线程named barrier，再加七段输出交接。即448次TMA请求和448次上述barrier执行/输出tile；这不是448次整GPU同步，也不是448次已测阻塞。EPILOGUE另将原8warp提交改为W2/W7两warp，增加独立输出槽及full/empty协议；机制能允许重叠，但没有量化的实际重叠比例。

3. **单个小投影和FFN2只有96输出tile。** 当前M224/N3072、M64×N128给出ceil(224/64)×24=96。在170个常驻CTA中，同一阶段最多96个CTA同时算这些输出，其余等待或（FFN局部）参与生产，末M块还只有32有效行。Self QKV联合队列有3×96=288个独立任务；其阶段已接近Torch，支持检查并行度，但不是隔离证明任务数独自解释差距。

4. **数值分段被实现为反复完整提交。** `../workspace/算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh:202`起，SM120小投影分9×K320+K192，十次启动分段mainloop、转存shared、合并上次global输出并写回BF16。`whole_layer.cu:376`与FFN2调用处则为七段K2048。必须保持每段BF16舍入和最后bias顺序，但数学契约不要求中间结果每段都往返global；将舍入后的部分和保存在register或独立shared槽是待验证的优化机会。不能只去掉舍入或改成全程FP32积累。当前计数是global访问请求，不等于每次真的访问HBM。

5. **整层资源约束没有被TMA消除。** 现行正常kernel为256线程、255寄存器/线程、约98KiB shared、1CTA/SM。这是整层容器的限制；TMA三个候选并未降低驻留率，所以不能说这轮退化由occupancy进一步下降导致。静态local/spill属性不是动态访存量。独立GEMM和整层内同一基元须分别测，才能拆出容器成本。

6. **两类GEMM实际warp配置不同。** FFN2为四MMA warp；Self-out/Cross-Q/Cross-out在`whole_layer.cu:198`的B14分派均进入`self_qkv_projection8`→`self_qkv8_tile`，已是八MMA warp，不能解释成它们也仅W3–6计算。附近仍有旧A100“两段K1536/其他投影4warp”的注释，不作为当前SM120实际分派证据；本次未改代码。

因此可确认主循环、交接、数值提交及并行度存在具体优化对象；尚不能给出各项硬件stall百分比或保证哪项是最大根因。没有可用NCU硬件计数器时，不编造带宽/TC利用率。

## 补充的关键原始trace：Torch确有跨CTA的K分段

00:01继续核查`../precision/results/partitions-data/{ffn2,selfout,crossout,crossq}-trace.json`的真实kernel事件，得到：

|原Torch kernel|输出tile（kernel名称）|grid|launch总CTA|线程/CTA|寄存器/线程|shared/CTA|
|---|---|---|---:|---:|---:|---:|
|Self/Cross输出投影、Cross-Q|128×128，K32四级|16×3×10|480|128|226|65536B|
|FFN2|128×64，K64三级|32×3×7|672|128|158|73728B|

这里480/672是launch总CTA，不是同时驻留数量。trace中的`blocks per SM`是总blocks/SM数的统计，不用它宣称occupancy。此前精度核验已关联十/七个K分段，现行自有却把这些段放在一个输出tile所属CTA内串行循环。也就是说，复现了BF16舍入契约，但没有复现跨CTA的分段计算组织。这是比笼统“新架构不快”具体得多的差异。

作为机制参考，官方[CUTLASS v3.9.2 Gemm kernel](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/kernel/gemm.h)先根据CTA的K坐标算局部MMA，之后才以semaphore按K次序提交：**串行归并不要求各段矩阵乘串行计算**。这份通用源码用于解释机制，不冒充cuBLAS内嵌kernel的逐行同源证明；本机实际launch信息以上述trace为准。当前没有新单因素结果说明这项能回收多少延迟。

这项证据使下一步优先级修正为：先把“原高效主循环＋K分段任务组织”做成局部对照，再隔离TMA收益。M32仍是备选，不再仅为填满SM优先尝试缩tile。当前实际版本未改变。

## 已找到的官方参考

NVIDIA CUTLASS GeForce Blackwell dense GEMM明确支持FP16/BF16、FP32累积、TMA输入及register MMA。只读取得文件的最近提交为`098de2a652cf8f00fd70b2df54051c7eccbb855a`（2026-09-22）；[固定上游源码](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)，本地引用副本和来源在`reference-review/`，未执行或导入。

其主循环也将下一fragment的shared→register读放在当前MMA前。借鉴这套访存/寄存器流水，不复制整个launch：该示例的warp数、资源分配、尾块、数据布局及默认数值契约与本项目不同。源码存在不证明其在M224和当前严格舍入下达到Torch性能。

设计依据另见[NVIDIA高效GEMM的两级软件流水说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)与[SM120资源上限](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)。TMA搬运可与计算重叠，不自动消除shared→register依赖或提高所有kernel效率。

## 建议的唯一下一步顺序（未实现、未启动）

1. **先验证局部split-K组织，保留原高效主循环。** 将FFN2的任务身份区分为输出tile与K段，段内保持原MMA次序，段间计算允许并行但提交仍按原BF16次序。先独立真实FFN2，再放入整层，比较同主体的容器开销；它是局部固定任务组织，不是完整DAG。若persistent CTA等待归并，必须证明前置段一定可运行，不能让所有常驻CTA等尚未领取的任务；若用临时partial缓冲与有序归并，缓冲/访存/同步须全部计入。先保持现有tile，不将Torch输出tile形状、TMA、warp配置同时修改。
2. **再同tile修TMA主循环，隔离搬运收益。** 保留四MMA warp、七段K2048、既有两槽与数学次序，恢复显式寄存器fragment预取；用相同任务组织对比cp.async/TMA，不再把精简主循环与新搬运打包比较。检查分段流水启动/收尾以及两warp提交是否抵消输入重叠。
3. **提交驻留与M32作为有证据触发的后续，非并行扩展。** 串行段候选可将BF16部分和留在register/shared；跨CTA split-K候选则必须另行处理跨CTA交接，不能假装共享普通shared memory。寄存器已接近上限，不能无条件加大数组。M32×N128给出168输出tile、无M尾块，但增加权重读取；三级K64输入60KiB＋单FP32输出17KiB＋BF16部分和8KiB≈85KiB只适用于相应串行段布局推导，未含metadata/对齐，未编译或测试。与split-K不同时展开尺寸网格。
4. **移植共同引擎，保持各自配置。** Self-out/Cross-out/Cross-Q同为M224/N3072/K3072，已有共同入口，最适合一起复用修复；保留其十段数值边界。FFN2为M224/N3072/K14336，可共享TMA/寄存器流水/写回框架而保留四warp配置。FFN1为M224/N14336/K3072，还含GELU且当前主8warp/尾4warp，不硬套同一tile或同一warp比例。

先使独立GEMM接近原Torch，再核对放入整层后的额外成本；仍以无插桩真实三KV整层净收益决定采用。对齐是目标，现有证据不能保证一定达成。GPU作业仍需新的具体有界实施安排；原账本余额不视为已启动。

## 本次归档与资源

仅CPU阅读、原始JSON统计、官方资料核验与HW更新；未修改优化源码/baseline/投机侧、未重置任何预算。原GRAIN与负结果保留。23:56:57只读见双A100均有模型计算进程，CPU五分钟巡检最近23:52:49；未改队列或统计成功率。
