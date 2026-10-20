# 先逐阶段对齐，再做跨阶段调度

状态：PLANNED_NOT_STARTED。源于本轮profile后用户最新反馈；没有新增GPU作业。本文件替代RESULT中“直接先改Norm/pointwise”的下一步执行顺序；原测量不变。

## 对照如何拆

|路径|执行内容|用途|
|---|---|---|
|A 原版阶段|冻结native中的对应计算，含同等bias、cast、layout及必要输出|阶段参考成本|
|B 自有阶段独立执行|复用optimized同一设备函数，用最小测量入口只执行该阶段|判断阶段原语和tile组织本身的差距|
|C 自有阶段嵌入整层|当前同一persistent完整层中的阶段|判断整层统一资源、任务领取、共享workspace和同步造成的增量|

B只是测量入口，不是新的生产实现或把整层拆回多kernel。仍维护baseline/optimized两份研究实现。B/C编译配置、资源和函数专用化差别全部记录；不能把所有差异一律称为global barrier成本。

## 覆盖顺序

1. 复用当前B14短/中/长KV真实fixture。先量化大项：cross K/V projection、cross-K RMSNorm、self/cross FA2、TABLE/GELU；随后补齐self/cross小M投影、FFN1/2、其他Norm、modulation、RoPE、residual、cache/layout。
2. A/B使用同一真实中间输入，包含shape/stride/dtype、权重、BF16分段舍入、所有有效KV/私有可见性。不得仅生成同shape随机数据。
3. 同一GPU、同一时段交错测量，准备与profiler分离，工作量和输出边界统一。A融合了bias/cast时不能把其成本漏掉；当前profile和native单kernel只作候选定位，不冒充已经完成这个矩阵。
4. CUDA事件/trace与设备阶段时钟分别标边界，记录kernel数、任务数、CTA/warp数、寄存器/shared/local编译数据。当前NCU硬件权限不足，动态计数继续UNKNOWN，不修改驱动或权限。
5. 按差距分类：B慢则修该阶段数据搬运/tile/归约；B接近A但C慢则优先修嵌入资源、workspace和队列/等待。每次一个因素，完整层与有效KV门复验，不能用局部更快掩盖整层回退。
6. 主要阶段差距收敛后，再实现context K/V就绪任务填尾波。完整层同输入延迟是最终系统工程指标；阶段差距相加只是诊断，不保证可实现同等总收益。

## 尚未完成与执行边界

当前A/C诊断已具备，B独立入口及严格统一边界矩阵未实现；因此各阶段正式倍数仍待测。本轮24block/150原语预算已终态，不继续消耗或重置。下一次GPU作业按确定入口单独登记有限调用、生命周期、存储及清理上限，不开无边界形状搜索。

整个范围仍单层、B14优先、原始action沿原算子。baseline/投机/原tile不动，四项质量权限false。RoboTwin固定42采集并行继续，不因本分析暂停。
