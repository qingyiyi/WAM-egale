# MegaMoE与当前SM120实现：源码差距核验

2026-10-25；当前采用S不变。外部参考固定DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`，原源身份清单为`../../transformer-tile-handoff-20261019/MEGAMOE-REFERENCE.json`（相对于本目录需沿项目索引定位）。本报告不把其他同名MegaMoE混入，也不引用其未经重现的速度作本项目收益。

## OBSERVED：七个关键区别

1. **硬件执行语义不同。** 官方BF16 kernel `sm100_bf16_mega_moe.cuh:726–813`由leader CTA的一个issue warp发2CTA UMMA，accumulator在TMEM；本S `cutlass_gemm.cuh:34–47`是SM80类warp MMA主循环，编译目标SM120，寄存器保存累加器。5090有TMA但不能因此照搬SM100 UMMA/TMEM角色比例。
2. **角色不只是1T/4M/3E。** 固定官方heuristics `mega_moe.hpp:123,222–223`为128 dispatch线程+128非epilogue线程+256 epilogue线程，共16warp/CTA；4个非E warp分别activation加载、weight加载、MMA issue、scheduler。官方kernel `:285–289`按角色寄存器预算，非E40/88、E208/160；我们的整层静态255regs/thread、100352B动态shared，普通寄存器/编译器资源不能由if分支自动分别缩减。
3. **输出流水槽不同。** 官方`:142–143,757–813,863–865`两级TMEM结果、两级TMA写回；我们W/A的单FP32结果槽在shared，与两级输入共占资源，四个MMAwarp先将累加寄存器写shared再交给E。官方TMEM释放/下一tile计算与上tile后处理可并行。我们已有局部full/empty不等于相同流水深度与服务速率。
4. **cluster不是同一种路径。** 官方使用2CTA UMMA及A multicast；我们以独立warp MMA加显式shared-to-shared bulk复制、credit/READY/ACK/EMPTY完成共享。多了软件交接，不能把两者都叫2CTA就预期相同收益。A2仅将握手从48降到24，并未减少逻辑总字节，而且去掉旧step+2预取的提前量。
5. **可调度工作量不同。** S FFN1源码`:754–813`静态120任务/170CTA，50CTA无FFN1tile；108主体内52块只有96/128有效行，另12个M32尾tile。W有448tile，但每CTA只有2–3个输出tile，短流水受启动/排空影响。官方按专家/行块配置分块，在多任务间反复调度并预热；不能只增加tile数而忽视重复加载与单位计算量。
6. **跨阶段ready调度未接入当前S。** 官方scheduler `:372–410`warmup后交错L1/L2，`:143–183`按L2的K块等待L1 ready位，带live-pool与前进约束。我们S按用户确认保留FFN1到FFN2的grid同步；当前不是完整MegaMoE式任务调度。这不表示立刻删同步：基础主循环及交接慢时更大调度器可能再添成本。
7. **原问题收益来源不同。** 官方融合专家dispatch/combine、线性层和激活，含多GPU通信/路由与ring管理；WAM没有这部分同等成本。官方`:978–1028`仍将激活通过TMA写global/ring后发布L2 ready，也保留组同步；所以“全部ACT驻留shared”“完全没有同步”都不是其成立前提。

## 本项目已观测的性能

S整层相对原Torch已快，最新独立正常轮约1.286/1.789/2.229倍；“我们不行”应限定为FFN1局部尚落后及新W/DSM增量无收益。S阶段FFN1约0.80倍Torch、FFN2约1.03倍，阶段插桩与正常完整层分列。

W整层慢S约0.8%；共享B慢49.89–60.62%；共享A慢21.70–26.48%；双K64交接A2慢25.92–31.73%。L分组/独立加载约等W，支持新增交接链是重要成本来源。软件等待包络彼此重叠，不能加总为纯DSM时间。没有硬件计数证据声称HBM瓶颈或HBM流量下降。

## UNKNOWN与下一最小证据

- S FFN1的CUTLASS主循环（包含输入读取与等待）和两次分半shared结果转存、GELU/输出、发布分别多少；此次只加观察器实测。
- 当前Torch FFN1在SM120的真实kernel、grid/block，与旧A100配置不能混用；CUDA activity trace单独记录，不采受限硬件计数器。
- 255regs/784B local属于资源属性，不等于动态spill流量；阶段等待也不能自动称屏障实现开销。
- 5090有限真实fixture不能证明完整WAM质量/最终E2E；还缺候选数1/3/7交互矩阵、完整模型集成及质量门。

## RECOMMENDATION

先对齐基础FFN1，按证据在任务粒度/尾块、主循环载入、寄存器到shared写回中选择单一改动；当前不扩DSM、不改七段BF16递推，不为复制MegaMoE外观采用更慢的实现。需要延后的是跨L1/L2就绪调度与ring，不是放弃persistent研究。

官方固定链接：
- https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh
- https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
- https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/jit_kernels/heuristics/mega_moe.hpp
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
