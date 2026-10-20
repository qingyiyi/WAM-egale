# 当前context128完整单层开销与源码解释

2026-10-20，北京。当前源固定自`f686769a327d01779c1dcdf09be6a4bdc934ffa6`；生产optimized数学/二进制未改变。本轮只做当前副本profile和同输入native完整层profile。24次完整block、150次独立Torch Linear；输出/中间量/有效KV有限逐位通过。不是新普通计时或机器人结果。

## 实测：瓶颈已经变化

三KV各3次完整插桩重放。数值为各阶段中位，包含阶段任务领取、阶段内工作和出口等待；whole计时从layout后到最终barrier，排除了之前workspace审计初始化。插桩改变二进制，不替代上一轮普通计时4.90/4.68–5.17/5.10–5.25ms。

|KV|插桩区间ms|GEMM合计ms/占比|cross K/V投影ms|cross K RMSNorm ms|self+cross attention ms/占比|
|---|---:|---:|---:|---:|---:|
|288|5.1825|2.9676 / 57.26%|2.0152|0.6543|0.5130 / 9.90%|
|1152|5.5822|2.9596 / 53.02%|2.0164|0.6512|0.9138 / 16.37%|
|2160|5.6915|2.7867 / 48.96%|1.8835|0.6057|1.3101 / 23.02%|

cross K/V是GEMM子集，不能重复相加。原70.70%–76.08%为旧实现，不能沿用。当前cross条件投影仍是绝对大项，但self Q/K/V/out等小M投影每段约0.06–0.076ms，同输入Torch Linear事件边界约0.07–0.08ms，已经不应把所有GEMM同等视为最大差距。

在同一fixture上，对冻结native完整层取3次CUDA kernel trace，通过明确kernel名和实际grid识别下列项目；每次完整输出和有效KV与未插桩当前实现再逐位比较。

|项目|当前persistent插桩阶段ms，短/中/长|native单kernel ms，短/中/长|
|---|---|---|
|cross K RMSNorm|0.6543 / 0.6512 / 0.6057|0.07235 / 0.06845 / 0.06842|
|调制表TABLE|0.2550 / 0.2548 / 0.2397|0.04243 / 0.03408 / 0.03360|
|GELU|0.1802 / 0.1811 / 0.1717|0.02160 / 0.01674 / 0.01683|
|self attention|0.2212 / 0.6233 / 1.0391|0.07200 / 0.18807 / 0.32908|
|cross attention|0.2847 / 0.2842 / 0.2650|0.10521 / 0.09251 / 0.09277|

这是诊断性对比：persistent阶段包含队列/barrier，native列仅kernel持续时间；不同二进制/进程的clock与插桩影响存在。不能把表中相除写成纯算术吞吐倍数或正式加速。它支持把norm/pointwise和FA2嵌入执行列为当前高优先级，不能独立证明某一条指令是根因。

证据：[原阶段记录](RESULT.json)、[阶段汇总](SUMMARY.json)、[native完整层输出门](NATIVE-PROFILE-RESULT.json)、[native组件汇总](NATIVE-COMPONENT-SUMMARY.json)。原始trace私有保留，公开只kernel事件摘要。

## 实现如何工作

入口`算子研究/wam_ops/optimized/whole_layer.py`绑定原参数/eps/cache，调用`csrc/whole_layer.cu:263`的单次cooperative kernel。当前真实B14输入hidden `[14,16,3072]`、context `[14,512,3072]`、24heads、head_dim128；输出hidden形状不变，KV按候选可见性处理。

单层按以下DAG执行：

`Norm1/调制 → self QKV/RMS/RoPE → self FA2 → 投影/残差 → Norm2 → cross QKV/RMS/FA2 → 投影/残差 → Norm3/调制 → FFN1/GELU → FFN2/残差`。

cross K/V只依赖context，源码当前仍在cross Q之后顺序做，没有提前进入其他阶段的空闲位置。

108个常驻CTA，每CTA256线程/8warp。`whole_layer.cu:73–86`中，warp0的线程0通过atomicAdd领取**当前阶段**tile，整个CTA随后共同执行；任务完成后再领下一tile。各阶段仍有grid.sync。这是常驻CTA执行器，不是每个warp独立领取任意DAG节点。

GEMM在`csrc/cutlass_gemm.cuh:13–99`：

- context展平M=14×512=7168，输出tile128×128、K64；每个K/V矩阵共56×24=1344个tile。
- action展平M=14×16=224，tile64×128，保留较小行tile；3072列投影共4×24=96个tile。
- warp3–6执行固定CUTLASS SM80三阶段cp.async/MMA；context每warp64×64，action每warp32×64。
- 计算结束，全部8warp按两个元素一组协作epilogue。保留原FP32 bias/partition加法顺序以及BF16分段/最终舍入。
- 此处大M的加载仍在计算warp内完成；warp1/2未成为与计算独立重叠的专职loader。

pointwise/cache在`whole_layer.cu:204–247`按2048元素领取，pointwise当前仍只有warp3–6执行。Norm在`:109–176`以完整3072维行归约：warp1搬输入，4个compute warp转换/归约，中间FP32放shared，再由warp7存结果；一个CTA一次只处理一行。

attention在`csrc/fa2_stage.cuh:72–126`中，warp3–6执行设备FA2的load/MMA/online-softmax/store，任务结束全CTA汇合。B14每个candidate仍独立Q16，正常路径Q tile128/KV tile64，每个self/cross共336个batch-head任务；不是Q112序列。历史与private suffix按原合法地址区分。

## 为什么仍有开销：事实与推断分开

OBSERVED：整个未插桩kernel仍255register/thread、98848B shared、1608B localSizeBytes、108CTA。native cross-K RMS kernel只有22register/thread、24B shared、128线程/CTA；TABLE为28register/256线程/0shared；GELU为16register/256线程/0shared；native FA2为255register/128线程、65536B shared。这里的资源来自编译/kernel metadata，不是动态计数。

HYPOTHESIS（由源码/资源支持，尚未隔离成因果）：整层统一资源预算令轻量norm/pointwise也承受大kernel的资源上限；它们没有利用本来可运行的更多CTA/warp。当前Norm反复shared转换与单warp写回、pointwise仅128个线程做事、逐tileatomic和barrier共同增加开销。FA2核心相同不代表外层执行效率相同：当前task调度/函数参数与spill、单CTA驻留、4个有效compute warp及尾波都可能产生损失。

UNKNOWN：NCU指标查询返回`ERR_NVGPUCTRPERM`，未改变权限/驱动，也没有启动NCU kernel replay。动态occupancy、stall、local-memory/spill流量、HBM/L2均未知。PyTorch trace里的`est. achieved occupancy %`是估算且部分高寄存器kernel显示0，**不作硬件实测证据**。当前stage时间也未单独分离global barrier与运算本体。

## 下一步优先级

本分析后用户反馈优先逐阶段对齐；因此先执行[同输入三路径对齐计划](STAGE-ALIGNMENT-PLAN.md)，再依下列测量线索优化。本轮未追加GPU作业，以下不是已经完成的原语因果归因。

1. **先优化Norm和pointwise执行组织。** 保留原128-thread Welford/RMS归约顺序与BF16边界，减少shared往返、协作向量写回；可比较每CTA两个独立128-thread行组/成组领行。pointwise让更多warp参与，采用向量IO并减领取次数。每次只改一个因素，用同输入状态门验证。不能直接换warp归约树来换速度。
2. **再优化FA2嵌入路径。** 长KV self-attn已超过1ms，先针对B14/Q16固定条件缩短活跃变量与device-call边界、检查参数/地址分支和实际tile执行；保持FA2累加/softmax/KV语义。降低register/spill只是候选手段，NCU未证明它是唯一根因。不要先盲改Q tile或引入跨candidate注意力。
3. **然后实现真实ready调度。** 把context K/V及其norm作为独立任务，在其他阶段尾波填空；使用epoch与release/acquire，消费者等待完整前置数据。仅调整函数调用顺序不产生重叠。FFN2仍需完整F14336输入，不能按不完整列tile开算。此前FFN ready的负结果保留，不能直接打开新路径。
4. 当前context GEMM继续保留CUTLASS主循环，后续可依据profile比较更合适的tile/epilogue；不再无依据铺开GEMM形状网格。最终必须与双方同等缓存的强baseline比较，cross K/V缓存带来的减少不作为新的调度贡献。

此顺序是基于当前诊断的推荐，不是已实现新异步或已保证提速。整个范围仍单层，四项科学/正式性能权限false，baseline/投机代码/旧tile不修改。
