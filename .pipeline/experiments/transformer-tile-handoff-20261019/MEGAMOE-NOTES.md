# MegaMoE源码参考与WAM/SM80映射

状态：源码审查完成，尚未实现或运行新的persistent kernel。官方来源为DeepGEMM，固定commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`；逐文件身份、许可证及永久链接见[源清单](MEGAMOE-REFERENCE.json)。只下载阅读，不安装、不升级CUDA、不执行外部代码。

## 观察到的实现

以下行号均对应上述固定commit，服务器缓存根目录为`reference/DeepGEMM/`。GitHub只公开源身份与永久链接，省略这份可重新取得的外部源码拷贝；本地原件保留。不能据当前main的后续变化改写本记录。

| 位置 | 实际机制 | 对WAM的启发 |
|---|---|---|
| `scheduler/mega_moe.cuh:83–116` | Linear1/Linear2等phase；TaskInfo携带m/n tile、局部专家、workspace槽位、有效M等 | 将任务改为layer/subgraph、CFG/candidate行区间、m/n/k块、buffer generation；不需要专家路由 |
| 同文件`:143–183` | L2按K块等待L1的完成位；producer发布、consumer获取ready状态 | FFN1/GELU某个列块完成后，可供FFN2对应K片段使用；不是要求整个FFN1全完成才开始 |
| 同文件`:15–79` | L1 warmup波数与live-pool容量计算，避免所有worker先等L2而上游未运行 | 先保证上游任务和workspace可推进；明确在途上界，不能仅写一个无限取任务循环 |
| 同文件`:230–259` | 双缓冲任务metadata及full/empty barrier | 常驻CTA/warp之间传递任务时，必须规定谁发布、谁消费、何时可复用 |
| 同文件`:344–369` | locality domain内原子领取task index，按m/n展开，支持有条件取其他域任务 | 可比较静态行走与动态领取；不预设原子队列一定优于静态、也不预设需要候选各一队 |
| 同文件`:372–410` | warmup后交错派发L1/L2，依据上游任务进度约束L2派发，含终止sentinel | 生产者/消费者任务共享资源时，把就绪条件和停止协议纳入调度器 |
| `impls/sm100_bf16_mega_moe.cuh:626/687/726/816/826` | activation加载、weight加载、MMA、scheduler、epilogue分工，循环取得任务 | warp角色专门化是具体执行结构，需要对各角色资源和等待时间测量 |

完整官方路径前缀为`deep_gemm/include/deep_gemm/`。[scheduler原文](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)、[BF16 kernel原文](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh)。

## 不能直接移植的部分

这条BF16路径使用SM100的TMA、UMMA、TMEM、双CTA cluster和寄存器重分配机制。它不是现成的SM80 BF16 persistent FFN。当前A100应使用支持的global/shared搬运、warp级Tensor Core MMA及适用的barrier；`cp.async`等可用能力以[官方Ampere说明](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)和实际编译为准，不默认TMA/WGMMA/UMMA可用。

常驻资源单位首先是CTA；warp在其内部承担任务角色。SM100的一条MMA issue warp不能直接变成SM80的等价代码，SM80的MMA tile通常需要一组warp协作。当前WAM FFN原语为4 warp协作的CUTLASS CTA，不允许让每个warp各自调用一个host GEMM。

MegaMoE的L2依赖公式含SwiGLU将宽度减半的布局关系。WAM是GELU，不减半，不能照搬`BLOCK_N/2`。专家路由、跨节点通信、低精度量化也不属于当前WAM子图。

## 最小映射与验证顺序：建议，未执行

1. **FFN1+GELU tile常驻调度。** 先复用已有有限逐位通过的设备原语，定义一个CTA连续领取多个输出tile的协议，固定K归约与两次BF16舍入。比较同tile的普通grid与persistent调度，单独计任务领取和尾波开销。
2. **FFN2设备原语门。** 旧CUTLASS FFN2逐位失败仍保留，当前正确实现使用host cuBLASLt；解决该门后，才能声称完整FFN为设备内tile子图。不能把host cuBLASLt放进kernel或用新的allclose门覆盖旧失败。
3. **逐级生产者消费者。** 首版先等一个行块的全部FFN1列块/GELU就绪，释放该行块的FFN2任务，保留完整K归约；之后才按证据研究K片段ready。细化后最终仍须覆盖14336维、保留归约顺序、FP32累积和原BF16边界。bias只加一次。
4. **warp角色与candidate-aware策略。** 明确load/compute/store/metadata角色，用SM80兼容原语实现。候选维度用来增加合法任务和选择共享数据，不能改变数学可见性。

初版建议保留完整global scratch而不马上做ring：B14的[224,14336] BF16激活为6,422,528字节，即6.125MiB，无法整块放入shared memory。ring需另证最小容量、每槽generation、消费者释放与无死锁；省workspace不是第一轮正确性的必要条件。

“投机使persistent更有效”仍是假设。现有FFN1 tile的M有效比例从32/64=50%变为224/256=87.5%，仅是几何推导；是否抵消任务管理、同步及更少CTA的代价需实测。先测1/3/7候选的shape×引擎矩阵，再判断是否存在交互，而不把普通M扩行当作新增调度贡献。

## 需要保留的反证

队列成本大于launch节省、资源占用下降、warp等待过长、普通库已经获得全部收益、子图加速不能传到block/verifier、严格数值路径无法保持，均是有效否定结果。这里没有新的kernel、30层、chunk或闭环性能数字。完整工作依赖见[工作大纲](WORK-OUTLINE.md)。
