# 用户提出的跨阶段tile填尾波方案

2026-10-20。讨论方案，未修改optimized/启动新GPU作业，不写成已经证明的性能结论。当前实际实现和负结果见[差距审查](MEGAMOE-GAP-REVIEW.md)、[最终单层](FINAL-LAYER-RESULT.md)。

## 判断

将tile作为任务，以常驻CTA和职责明确的warp执行，根据输入就绪领取工作，用独立分支填补阶段尾部，是与MegaMoE相符的调度思想。需要把“等一轮全部做完”改成**每个CTA完成当前tile即可选择下一个ready任务**。一个GPU并没有要求这108个CTA按统一轮次进度；总tile数量大于108也不说明当前同时就绪的tile足够。

依据固定官方源码：DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d` 的[scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)、[warp分工](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh)。可借鉴依赖/任务/角色流水，不能把SM100的TMA/UMMA/TMEM和动态寄存器分配照搬到SM80。[A100架构约束](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)

## 真实DAG中的机会

```
hidden -> Norm1/modulation -> self Q/K/V -> self attention
       -> self out/residual -> Norm2 -> cross Q -------------------+
context -> cross K -> K norm -----------------------------------+-> cross attention
context -> cross V ---------------------------------------------+
       -> cross out/residual -> Norm3/modulation -> FFN1/GELU -> FFN2/residual
```

cross K/V只依赖本层输入context及本层权重，在block入口已具备输入；可以和self分支交错，不必等Norm2。cross Q不能提前到Norm2之前。cross attention仍需合法Q/K/V及各自完整归约；FFN2仍按原数学次序消费所需输入。跨候选private KV不共享，不改变attention可见性。

注意cross K/V在cross attention前必须已完成，不能到更后的FFN尾波又拿未完成cross K/V填空。后半段需要FFN自身的ready生产/消费；当前行组方案虽已有交接，实测反而更慢。

## 当前B14几何量

仅对应当前64×128输出tile及M224/action、M7168/context，不是实测利用率。

|节点|输出tile数|若假设108等速CTA的示意尾部|
|---|---:|---|
|单个action D→D投影|4×24=96|12个CTA无同阶段任务|
|cross K或V投影|112×24=2688|24个满轮＋96；自身尾部比例很小|
|FFN1|4×112=448|4个满轮＋16|
|FFN2|4×24=96|输入K14336，不能视为与K3072的tile等时|

可用独立cross K/V填早期投影空档，优先关键路径，独立工作作后备。不能为了维持忙碌把所有CTA派去长独立任务，反而推迟刚ready的self关键依赖。任务粗细、权重L2局部性与队列开销一起考虑。

## CTA内部角色

当前已经是warp0调度、1载入A、2载入W、3–6计算、7写回。差距不在角色名称，而在它们是否可以流水推进：当前仍频繁CTA同步；FA2计算warp自行load/MMA/softmax/store；多数阶段外还有grid同步。

合理结构是scheduler warp给**本CTA的协作warp组**发tile描述符，loader/compute/epilogue通过有界buffer full/empty交接。不是同一CTA的每个warp各拿任意不同算子任务。不同CTA可以执行不同ready算子，但旧条件分支内的grid.sync不能原样保留，否则异构阶段交错可能死锁。依赖就绪须有精确计数/epoch/release-acquire；没有ready consumer时必须允许producer继续，不让所有CTA自旋堵住producer。

## 速度结论与最小下一步

**能减少尾部空闲和某些阶段等待，尚不能保证整体更快。** 当前B14仍慢原Torch3.96–5.02倍，旧ready负收益2.5%–2.7%。仅改善投影96/108这种几何空档，不能解释或消除数倍退化。应同时修复当前GEMM/写回/资源效率，区分静态寄存器、共享内存和实测stall。

对照必须包括双方相同cross-KV缓存。缓存命中时K/V计算已经不存在，不能特意重算来制造可调度任务；普通cache收益与新增ready调度分开。保留强B14 cached/uncached对照，不只对慢旧实现。

最小建议：同一GEMM与同一数值边界，对比阶段barrier版与“self链关键优先＋已就绪cross K/V后备”版；量出真实CTA idle/wait、队列和整体耗时。同时检查有限fixture输出/有效KV逐位一致。结果决定是否扩到attention/FFN，不预设全部融合就会赢。本条是建议，尚未启动。
