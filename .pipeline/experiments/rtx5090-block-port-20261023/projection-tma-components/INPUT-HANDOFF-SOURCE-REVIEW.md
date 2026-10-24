# TMA发起与输入交接：源码核查和最小改法

本页是2026-10-24的CPU源码分析，不是新增GPU实验结果。用户已同意先处理输入发起/交接，暂不引入后处理warp分工。当前采用版本仍为SPLITK-M64-FFN2；以下TMA源码是此前未采用的候选，不是当前默认执行路径。

## 已观察的实现

源码：[whole_layer.cu](../projection-tma-continuous/candidate-source/csrc/whole_layer.cu)。

- 99–103：两个K64输入槽；每槽A64×64 BF16和B128×64 BF16，共24KiB；双槽48KiB。分段FP32结果有独立区域。
- 180–194：thread 0执行expect_tx(24576)，发起A/B两次TMA。每输出tile共48个K64块、96次A/B请求。
- 212–215：先装入块0和块1。计算j时已有j+1在另一槽；覆盖当前槽时装入的是j+2。
- 224–228：thread 0轮询当前槽ready barrier，CTA同步后消费。
- 235–248：八warp全部做MMA；每个K64块含四个K16 group，使用两个寄存器fragment预取。group2迭代中已加载最后一个group3；之后剩余transform/MMA使用寄存器fragment。
- 250–254：当前实现直到四个group的MMA全部执行后才进行读者完成CTA同步，再由thread 0发起j+2。这是保守的释放边界。
- 256–303：十个数值段仍各自清零FP32累加器，再按原BF16次序合并；9段K320加末段K192。该顺序不属于本次改动。
- 337–347：阶段输入的generic→async proxy发布和原grid同步保留，不能为减同步任意删掉。

## 待验证的解释

现有实现已经有双缓冲，但不是独立生产者warp流水。thread 0所属warp也参加MMA；输入ready等待、读者交接和TMA发起仍可能落在关键路径。原软件分量观察不能独立证明硬件TMA发起慢或带宽不足；约9微秒发起区间含插桩和记账，不等于纯指令成本或传输时延。

可以验证的最小机会是：所有warp完成最后一次shared→register读取后，当前shared槽就可覆盖，不必等全部寄存器MMA执行结束。该条件必须对八个warp同时成立，不能依据thread 0自己的读取完成放行。

## 第一个最小候选

将现有第250行的读者完成CTA同步移至group2中最后一组fragment加载之后。在这个同步之后由thread 0发起j+2，随后继续原group2/group3的寄存器transform/MMA。保留ready等待与其发布同步，不新增输入槽、不改变warp比例、tile、K数值段、累加次序或后处理。末尾step+2<48及phase保持。

这是移动一次现有屏障和发起位置，不是删除全部同步。须先核对CUTLASS迭代器/transform确无后续shared读取，并检查编译器生成的寄存器/局部内存变化，再进行真实输入验证。若thread 0发起指令的代价超过可隐藏的等待，候选仍可能退化。

## 后续验收与停止

先比较三档真实B14输入的全部已捕获中间量、输出、有效KV、输入/RNG及48/48/10/48计数，再做两个独立进程的无插桩完整层比较：当前A、原TMA、候选、原Torch/guarded，同等eager和预计算cross KV。只有候选相对当前A有可重复净收益才考虑采用。

实施时另登记有界作业，不复用已过的绝对deadline，原累计剩余1813.587秒不变；预留清理。若闭合候选，必须更新并在对话打印完整阶段表，未重测阶段明确标历史。该源码分析未启动GPU、未改变采用代码、未产生新性能数值。

## 后续讨论：专职TMA warp与full/empty交接

用户进一步提出由专职warp提交TMA，并询问mbarrier能否替代频繁整CTA同步。这是布局讨论，不将具体warp数提前登记为确认决定。

当前第225行已由thread 0等待ready mbarrier；第228行将就绪状态交给全CTA。第250行是读者完成再利用同步。拟议的生产者/消费者协议应为每槽full（TMA完成）与empty（所有消费者最后一次shared读取完成）两条信号，而非仅将现有__syncthreads替换成同人数mbarrier。必须保证phase、消费者到达次数、首次槽状态、末尾未完成事务及acquire/release/async proxy顺序。整层阶段grid同步与结果交接不属于可直接删除范围。

固定DeepGEMM参考1fc9c66975ee322d628b3288534ca0e9ce12a64d的sm100_bf16_mega_moe.cuh中，657–681和691–724为A/B专门化TMA生产者，均先等empty再提交；778–780消费者等待full。这支持借鉴缓冲协议，不支持照搬SM100 UMMA/TMEM到5090。

现有小投影的八MMA warp按w%2与w/2映射64×128输出，抽走一个需要重排输出覆盖。直接加第九warp也存在资源问题：按现有255寄存器/线程估计，288×255=73440，超过SM120每SM的65536个32位寄存器；这是沿现有编译属性的资源估算，不是新九warp二进制实测。后续须先检查编译资源、局部内存及全层256线程假设，再选择布局，不能预设独立warp免费或必然快。

官方核查入口：CUDA13.0 PTX ISA（mbarrier.init/test_wait要求SM80，try_wait和事务扩展要求SM90起）；CUDA13.0 Programming Guide的producer/consumer双缓冲；Blackwell Tuning Guide的SM120寄存器资源。本讨论没有新编译、GPU作业或性能结果。

## 四/六MMA warp与后处理专门化的讨论

本节只记录2026-10-24用户提出的布局问题和源码分析，不视为改变既定八MMA实验的确认，也不启动候选。

固定参考MegaMoE不是没有同步：sm100_bf16_mega_moe.cuh第192、241行有cluster同步，第386、559、856行等有参与组同步，第1015、1143行有epilogue组同步；主循环通过full/empty交接并使用warp同步。它没有在每次输入搬运后将所有角色统一拦在__syncthreads上，并不代表整个kernel没有CTA范围同步。相关API还可能封装底层barrier，不能以未搜到__syncthreads字符串判断无同步。

局部热循环可以用按缓冲的mbarrier协议取代整CTA集合；mbarrier自身仍可位于CTA范围。初始化、公共workspace复用和整层grid阶段依赖需另外证明，不能一并删除。

优先讨论四MMA的原因是当前64×128输出自然可分为2×2个32×64 warp输出块。现八MMA为2×4个32×32。六MMA不是硬件不支持，而是当前映射不能直接改计数；需重切形状或不均匀分配，暂不引入。

一个待验证的八warp布局是W0提交TMA，W1–W3做epilogue，W4–W7做MMA。这仅是方案，未实现或比较。四MMA对同一输出tile的每线程FP32 accumulator数量是八MMA的两倍，必须检查寄存器和spill；不能宣称少warp必然降寄存器或更快。

SM120的mma.sync累加结果仍在MMA线程寄存器，专职epilogue无法直接读别的warp寄存器。需要MMA把结果写shared，经result_full通知epilogue，完成BF16有序合并后经result_empty允许复用。单个结果槽也可让下一数值段的寄存器计算与上一段epilogue重叠，但MMA再次写shared前必须等槽空；仍保留原十段合并与最后bias次序。

不能直接增加两个完整FP32结果槽：48KiB双输入+2×34KiB结果=116KiB，已超SM120每CTA99KiB，尚未计barrier/对齐。第一版优先单结果槽，不展开跨tile大流水。

CUDA13.0 PTX9.0的setmaxnreg目标列表明确包含sm_120a/sm_120f。此前255×线程数是沿当前普通sm_120编译属性的静态估算，不是证明架构无法进行角色寄存器重分配。若后续采用该特性，须先在实际5090工具链验证架构目标、四warp组一致调用与资源，不将文档支持冒充本项目已启用。四MMA连续占一个warpgroup便于研究这个机制，当前未编译或执行。
