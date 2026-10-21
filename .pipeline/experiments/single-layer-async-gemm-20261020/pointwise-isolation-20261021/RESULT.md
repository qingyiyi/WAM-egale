# 逐元素与FFN2有限对齐结果

更新：2026-10-21T09:13:44.589629+08:00。用户要求先对齐两处，再汇总远端。全部本轮作业已终态。

## OBSERVED：先分清数学定义与实现

“逐元素”表示元素之间不归约，并不表示CPU式串行。真实Torch GELU的profiler kernel为
`vectorized_elementwise_kernel<4,...GeluCUDAKernelImpl...>`：四元素向量化，128线程/CTA，3136个CTA。
原Torch调制表达式是`scale_shift_table[None] + temb.float()`，这次实际分成dtype转换和加法两个kernel。
此前native代理GELU是256线程/CTA的标量索引实现，不能把native代理当真实Torch的向量化证据。

当前自有数学体独立剥离：保留FP32公式、BF16舍入，分别测全网格标量、固定108 CTA标量、全网格vec4；
与同一真实输入的Torch/native/原persistent阶段交错测量。不是smoke、随机输入或机器人episode。
每个独立算子计时是热输入CUDA Graph、5次调用图、8次采样；profile另跑，不混入正式计时。

|中KV，两个独立进程，µs|真实Torch|native代理|自有全网格vec4|自有固定108 CTA标量|旧persistent阶段|
|---|---:|---:|---:|---:|---:|
|GELU|10.44–11.19|15.87–16.56|10.85–11.75|43.83–44.94|53.15–53.88|
|调制表|45.77–45.77|30.52–30.62|18.53–18.84|71.78–71.78|72.29–74.96|

独立GELU vec4在中KV约比Torch慢4%–5%，三KV/两进程范围约4%–6%；所有比较均有限逐位通过。
调制表vec4更快包含融合原Torch的cast+add、减少寻址和向量搬运，不能称为单纯向量化独占收益。
固定108 CTA标量即使无队列、无shared预留仍慢，支持并行布局/吞吐限制，**不能把全部差距归因于同步**。
全网格pointwise只需14/15寄存器、0 shared；原persistent阶段预留98848B shared、108CTA。
本轮没有NCU实际带宽/动态occupancy证据；profiler中的occupancy估算不是实测。

## OBSERVED：FFN2写回与分段

输入`[224,14336]`，权重`[3072,14336]`，输出`[224,3072]`。原实现每个输出tile依次算两个K7168分段：
第一段FP32累加后转BF16保存；第二段先加bias，再加第一段BF16值，最后转BF16。
写回没有发现数值错误。不能去掉这个舍入边界来制造“精度更高”版本。

相同输入比较M64串行、M128串行、M64 shared暂存第一段和M128两段并行。
后者仍是一个独立kernel，96个CTA计算48个输出tile×两个K分段，通过一次网格屏障保持有序提交。
只改变FFN2内部组织，没有实现跨Transformer阶段ready调度。

|中KV，µs|进程1|进程2|
|---|---:|---:|
|Torch|109.57|109.67|
|旧persistent独立阶段|150.63|150.63|
|独立M64/串行K|138.55|138.44|
|独立M128/串行K|189.24|189.03|
|M64/第一段放shared|139.16|139.06|
|M128/并行K|114.59|114.28|

第一轮减少全局写回改善约1%–2%，复测没有稳定收益，不能解释原约38%的差距。
M128串行只有48个CTA，直接放大tile反而慢；同M128改两段并行后，三KV/两进程比Torch约慢4%–6%，有限逐位一致。
Torch profiler实测grid为[24,2,2]、block128；只凭z=2不能断言库内部全部归约实现，未冒称已取得cuBLAS设备源码。

## 完整层：保留向量搬运，撤回FFN2整层并行

两种集成版本分别通过9签名（B2 action/B14 action/B2 video×三KV），输出、23共享中间量、
18个有效KV状态转移、输入/RNG、重复结果有限逐位通过。并不覆盖所有输入或闭环生产。

向量搬运保持point tile2048、108CTA、全部阶段顺序和同步，整层稳定改善2.7%–3.1%。
之后FFN2并行嵌入整层，短KV慢2.26%/4.05%，中KV慢约0.26%，长KV无改善；因此已撤回。
负结果及测试源在`full-ffn2-parallel/tested-source`；恢复收据证明当前为已验证的full-vector源与二进制。
新增FFN2独立收益**没有传到整层**，不能声称已经完成FFN2整层对齐。

当前保留版本的完整层计时如下。每行使用同进程交错顺序；三KV绝对时间不能脱离频率状态直接横比。

|进程/KV|native ms|本轮前 ms|当前 ms|当前/native|改善|
|---|---:|---:|---:|---:|---:|
|1/short|2.6342|3.8636|3.7453|1.422×|3.06%|
|1/middle|2.4062|3.5626|3.4617|1.439×|2.83%|
|1/long|2.8486|3.8100|3.7066|1.301×|2.71%|
|2/short|2.6250|3.8559|3.7345|1.423×|3.15%|
|2/middle|2.4000|3.5575|3.4524|1.438×|2.96%|
|2/long|2.8386|3.8127|3.7030|1.304×|2.88%|

完整层是uncached/eager边界，强cached/Graph、30层、chunk与机器人E2E不在本轮。
独立阶段与整层的计时边界不同，不把两者直接相加或以差值命名纯barrier成本。

## HYPOTHESIS / UNKNOWN / RECOMMENDATION

- 已观察：独立FFN2并行版本238寄存器、0 local；整层统一255寄存器，local从1464升到1488B；
  shared仍98848B、108CTA、1CTA/SM。后者静态变化与退化同时出现，不能据此证明动态spill是根因。
- SASS核验：独立vec4实际出现BF16的LDG.E.64/STG.E.64，以及调制表FP32的LDG.E.128/STG.E.128。私有工具目录缺cuobjdump的首次CPU探针失败保留；随后使用已有系统工具只读反汇编成功，没有安装或改动工具链。
- 未知：新增FFN2在整层内的精确阶段代价、其他阶段受编译布局影响、动态spill/HBM/L2和barrier占比。
- 下一步仍先对齐：固定整层资源条件比较FFN2，隔离函数编译/寄存器与屏障成本；不新增跨阶段ready。
  GELU/table已保留安全收益，不继续盲目放大tile。

## 原始证据与实现

- [所有独立输入/进程/原始采样](ISOLATED-STAGES.csv)，JSON为`process1/2-RESULT.json`及`ffn2-parallel/process1/2-RESULT.json`。
- [所有完整层测量](FULL-LAYER.csv)，两个集成目录保存原始采样、九签名门、状态和失败/撤回决定。
- `kernel-summary-*.json`保留实际Torch/native profiler kernel与几何；完整trace留远端，不与计时合并。
- 活动源码：`算子研究/wam_ops/optimized/csrc/whole_layer.cu`的`Point4/point`，FFN2仍为原`cutlass_gemm.cuh`串行路径。
- 独立FFN2诊断：`ffn2-parallel/isolated_cutlass.cuh`与`isolate.cu`；不是生产入口。
- baseline、LIBERO投机、原tile、原FAILED/retention及四项false均未改变。
