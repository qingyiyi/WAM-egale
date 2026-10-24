# 连续TMA候选源码核对

CPU_SEQUENCE_PASS，尚未GPU通过。仅根代理源码自审，不冒充独立审查。

三个小投影保留原96输出tile队列、8MMA warp、warp布局32×32及每段零起点；原十次逐BF16 epilogue代码从实际self_qkv8_tile提取，FP32相加/bias顺序未改。原FFN1/FFN2、SelfQKV联合队列、attention、norm/RoPE函数逐字相同，见CPU-AUDIT.json。

输入ring只有两槽：每槽A8KiB＋B16KiB；结果独立34KiB。barrier每tile初始化一次，48次消费对应phase逐slot交替；十个数值边界不重置ring。当前输入全部warp读完之后才覆盖，所有48次TMA完成/消费后才inval。投影入口每个参与线程执行async global proxy fence并通过原grid入口同步发布上阶段generic写入；权重不可变。

仍有每K64两处CTA同步，不能宣称没有等待；八warp后处理不与MMA重叠。内层寄存器两组fragment在下一组加载后再做当前组MMA；跨K64完整寄存器预取并未等同原CUTLASS所有调度。真实数值、性能和编译资源都必须测。

六个TMA descriptor在每次forward构建，其CPU成本保留在实际完整入口中，不假设免费。没有改baseline或缓存契约，不把性能结果继承给生产。
