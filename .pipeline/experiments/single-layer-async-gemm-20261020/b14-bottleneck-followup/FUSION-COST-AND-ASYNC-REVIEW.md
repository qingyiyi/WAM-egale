# 两倍阶段目标、复用范围与异步现状

本次只读代码、旧编译器日志及远端终态，没有新增GPU调用、构建或修改算子。日期：2026-10-20T18:33:05+08:00。

## 源码与编译器已经确认的事实

1. `cutlass_gemm.cuh:49–57`调用固定CUTLASS设备MmaMultistage并存下累加器；`cutlass_gemm_codegen.py:12–16`只改两个主循环同步为128线程named barrier。是复用核心计算，不是完整照搬经过调优的GEMM kernel；GEMM是CUTLASS核心，Torch Linear当前用cuBLAS，两者也不是同一二进制。
2. `cutlass_gemm.cuh:60–69`仍使用自有写回：四计算warp将64×128 FP32累加器放入共享内存，warp7的32线程循环处理8192元素，做原bias/分段BF16顺序，再全CTA同步。每个线程最多256元素。这不是CUTLASS完整epilogue；读写布局、并行度、流水均需重新验证。
3. `whole_layer.cu:371–384`返回实际资源：255register/thread、256线程、active blocks/SM=1。每SM只驻留8warp，相对SM80硬件64warp是12.5%的驻留warp上限；不是NCU测得的achieved occupancy或SM利用率，不能等同剩下87.5%算力闲置。大M计算仅warp3–6；warp1/2空闲但仍属于占资源的CTA。
4. 原编译日志明确`cutlass_gemm_tile`为248B stack/244B spill stores/244B spill loads；whole_layer入口80B stack/44B stores/220B loads。见STATIC-SPILL-EXTRACT.json。能确认生成spill访问，不能直接转成实际总字节、延迟或根因百分比。此前localSizeBytes1536本身也不等于动态spill流量。
5. `whole_layer.cu:72–85`每阶段begin执行grid.sync、每tile全局atomic领取并CTA同步。pointwise按128元素一个任务：B14 modulation table共4,128,768元素→32,256次有效tile；GELU3,211,264元素→25,088次有效tile。数字是源码推导，非新profile。尚没有去掉大多数全局中间张量。
6. `whole_layer.cu:266–306`明确按完整stage推进；cross K/V在self-attention之后才计算，尚未提前填尾波；Q/K/V各自独立stage。不能将整个执行称为已实现异步DAG。

## 异步覆盖

- 已在当前默认路径：CUTLASS tile内部三级cp.async搬运/计算流水。这不是跨算子就绪调度。
- 曾实现：`ffn_pipeline.cuh:45–111`，FFN1+GELU完整64行组所有列ready后，用release/acquire让该组FFN2先行；子图前后仍有grid.sync。
- 前版实测ready更慢1.2%–6.0%，默认False。新CUTLASS二进制该支路尚未重新验证，所以不能恢复或声称当前通过。
- 尚未实现：整层ready队列；cross-K/V提前；self/cross的细粒度依赖交接；各warp专职流水贯通；全层阶段全局同步移除。不能直接删除grid.sync：必须建立数据ready、可见性和workspace复用协议。

## 下一阶段工程目标及顺序

用户明确先将B14短中长KV耗时控制到同轮Torch两倍以内。本次同轮中位换算：短当前7.9375→目标≤5.3074ms（约再降33.1%）；中8.1437→≤5.8866ms（27.7%）；长8.6423→≤6.7702ms（21.7%）。数字只用于当前样本规划；验收必须重测同轮baseline，不能固定旧ms，也不是承诺已能达到。

1. 先同输入分离GEMM自身与嵌入开销：独立成熟mainloop+epilogue、当前device helper、完整层中该阶段。独立kernel只作诊断，不改变最终单kernel目标。新预算下同时测新阶段和spill相关计数，当前没有这些新数据。
2. 优先改自有低效适配：保持BF16顺序的协作/向量化epilogue，B14编译专门化和设备调用/寄存器生命周期；不直接限制寄存器数以免制造更多spill。
3. pointwise按更大连续行组领取或随producer处理，减少数万次小ticket与重复全CTA同步；保留原舍入边界。普通顺序阶段先接近两倍目标，必要的异步作为可测手段，而非预设只有去barrier能解决全部差距。
4. 再引入跨阶段ready调度，先独立cross-K/V填尾波及有限FFN交接，证明就绪/数据可见性/槽回收。对强cached baseline公平；缓存命中时不重算K/V制造可调度工作。

这些为分析建议，未实现/测量不写完成。目标不放宽精度、四质量门或原预算；旧1497/1500和截止不重置。没有把替换前70%–76% GEMM占比当成新CUTLASS版占比。

## RoboTwin现场

本次只读现场确认GPU compute列表空、first6-3所有owned身份均不存在；没有自动接续目录。固定专家114/114，Target六FIT身份5成功1完整失败（open_microwave），83policy、6557teacher；实际1520/罚分1500分列。TF collector/训练/DEV链尚未完整实现，不能称投机迁移已经完成。原准备18:20北京和first6 17:46:42截止均已过去，只核验终态，不重启或重置旧账本。详见FIRST6-TARGET-RESULT.md。

来源：[NVIDIA Ampere资源说明](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)、固定CUTLASS3.5.1源码、现有cutlass-build-r2.log和CUTLASS-GATE/TIMING结果。
