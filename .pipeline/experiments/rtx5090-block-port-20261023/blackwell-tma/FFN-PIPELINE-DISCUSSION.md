# FFN双层流水讨论：借鉴MegaMoE，适配SM120

状态：DISCUSSION_ONLY；用户本轮明确先讨论。未改数学源码、未编译或启动GPU；当前保留GRAIN-M64-FFN2，原预算/科学权限不变。以下具体设计属于建议，未登记为采用事实。

## 原文核验

沿项目既有外部固定提交`1fc9c66975ee322d628b3288534ca0e9ce12a64d`核对，不用最新main覆盖该版本判断：

- [BF16 MegaMoE计算体](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh)：独立activation/weight TMA角色、MMA issue、scheduler、epilogue循环；输入full/empty及TMEM full/empty分开；L1写出激活后发布L2依赖。该实现用SM100双CTA UMMA/TMEM等，不能当作5090可直接调用的内核。
- [依赖与调度](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)：L2KBlockDependency按当前K块等待对应L1输出；有L1预热/防死锁及live pool容量计算。其SwiGLU会减半中间列，本项目GELU不减半，不能照搬BLOCK_N/2。
- [CUDA13 TMA](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)：异步代理可见性、事务完成和槽复用是不同义务。
- [Blackwell资源限制](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)：SM120每CTA最多99KiB shared。本轮资源预算使用实测sizeof和编译资源，不能用SM100的227KiB。

## 当前源码事实

FFN1主108个M128/N256任务使用8个MMA warp，尾12个M32/N256任务使用4个。主/尾都先完成K3072、bias/BF16，再GELU并写ACT。当前FFN2是96个M64/N128输出任务，4个MMA warp、7个K2048段；须等所属M128行组的全部14336列ACT就绪后才开始。当前只在不同生产行组之间有重叠，尚非按2048列片段消费。

`action_ffn1_produce`静态派发120个生产任务；当前170个常驻CTA全部可驻留，且每个CTA先完成自己的生产工作再进入消费阶段。这是可沿用的有限防死锁基础，不能外推任意shape/CTA数量。GELU不能作用于FFN1尚未完成K3072的部分和。

## 建议的两个层次

1. FFN2 CTA内部先以已通过的M64/四MMA warp为锚。TMA producer与MMA consumer各自循环，通过每槽empty/full及phase交接；只参与者同步。保留必要的四MMA warp共享内存读写同步，不能直接删除所有barrier。full表示数据到齐，empty必须等最后shared读者结束。先隔离输入流水收益，再单独考虑epilogue重叠。
2. 跨CTA让FFN2按原7个K2048段顺序等待ACT片段，避免等待全部14336列。普通M128生产行组的一个2048片段包含8个N256 FFN1 tile；最后生产行组的末片段含4个M128 tile＋12个M32尾tile，不能一律数8。首版可保守按M128生产行组发布，两个M64消费者共用就绪信息。每个输出tile继续按原p=0..6顺序做BF16分段合并，bias/残差位置不变，不用atomic乱序相加。

ACT初版仍保留完整global workspace，供24个输出列块消费，不重复计算FFN1、不假定不同CTA能直接共享普通shared。写ACT、发布ready、消费者acquire和TMA代理可见性须正确衔接。仅有ready计数不构成完整内存协议。全部生产任务先已派发/有保障进展，避免消费者占满CTA而等待尚未执行的生产任务。当前120生产任务/170常驻CTA可用静态预分配约束，暂不引入通用DAG。

## 一个待评估的FFN2角色分配

W0任务metadata/派发，W1发A/B TMA，W3–6保持MMA，W2/W7后处理。该表是建议，不是已实现或最优比例。FFN1主tile当前8warp全做MMA，没有四个闲warp；第一阶段继续其原计算体，不把FFN2配比强套到FFN1。

5090上的累加器在MMA线程寄存器中，epilogue warp不能直接读取。要交接必须先由MMA warp写独立shared输出槽，再发布full；epilogue按序读/提交后发布empty。输入与输出同时活跃时不能继续用同一union覆盖彼此。可先只给单个FP32输出槽：当前段p后处理时，MMA用寄存器计算p+1；只有下一段写输出槽前需要等待empty。这样保留七段提交顺序，不要求两个输出槽。

对于M64/N128/K64：
- 每输入槽A为8KiB、B为16KiB，两个输入槽共48KiB；
- FP32输出槽64×136为34KiB；
- 两输入槽＋单输出槽约82KiB，尚未计barrier/对齐等，需检查整层最大live集合；
- 若两个完整FP32输出槽则约116KiB，超过SM120单CTA99KiB。

当前整层97.75KiB来自各阶段最大存储union；82KiB不是在97.75KiB之外直接再加，也不代表整层最终会降到82KiB。寄存器/局部内存和编译结果还须测。

## 讨论中的实施顺序与未知

建议先同M64形状隔离TMA producer/consumer输入流水，再检查单输出槽epilogue交接，最后增加K2048片段ready；每次与保留GRAIN及强guarded同条件对比。FFN1的TMA改造在其瓶颈明确后另议。减少等待不保证足够重叠：现有120生产任务可能很快在同一波结束，K片段ready的收益须看真实完成时间分布。两epilogue warp也可能变成新瓶颈，不预设比原8warp写回快。

这轮没有新性能数据，TMA负结果及当前GRAIN选择保持。用户已确认的参考偏好和“先讨论”范围已合并到Maintain；本页具体配置和推断没有提升为长期约束。
