# 当前完整单层阶段表：六投影8warp＋Self QKV联合队列

更新时间：2026-10-23T02:03:36.590751+08:00。当前源码`fb650be4d78b`，完整身份见[SELECTED](SELECTED.json)。B14×Q16×D3072，24 heads、head_dim128、FFN14336。单个persistent kernel启动108常驻CTA，每CTA8warp/256线程；112384B shared、255寄存器/线程、静态local1016B、1CTA/SM。总数学tile仍3408/3480/3408，grid同步27→23。

**原Torch/自有完整单层warm Graph：1.2967× / 1.8469× / 2.9910×**（短288/中1152/长2160 KV）。最终同轮直接对比四臂，不乘旧轮次收益；相对用户上版延迟降2.211% / 1.995% / 1.841%。双方同等预计算最终Cross K/RMS和V，Torch静态dense history；这是单层warm测量，不能替代30层、chunk或机器人E2E，也不是最强guarded增量。

所有阶段已按当前版本两进程重测。原Torch阶段为原模块表达式在对应真实中间输入上的独立热Graph，自有为完整kernel中CTA0进入/退出区间，包含队列和同步；缓存上下文不同，不可相加当完整层耗时或精确可省时间。测量副本资源相同、完整层扰动≤0.197%，对应输出/中间值和输入/有效KV/RNG有限逐位通过。

QKV现在是一个联合阶段：Torch也一起回放三次独立投影，分别核对三份输出。不把联合耗时拆成三份单独测量，不直接拿旧单投影热缓存比值解释增益。阶段时间戳约1.024μs量化。

G表示本行grid同步次数，不含CTA内部barrier。tile是逻辑输出任务，不能与108常驻CTA或独立kernel数混用；比值大于1表示自有更快。

|阶段|做什么 / 每tile工作量|tile数（短/中/长）|CTA内warp分工|G|原Torch/自有（短/中/长）|
|---|---|---|---|---:|---|
|可见历史layout|CTA0按每轮256物理slot压紧有效历史；实际pool由输入决定|—|仅CTA0的W0–W7|1|N/A（准备/已融合/缓存/无对应数学工作）|
|调制参数表|不再生成m×6×D表，在消费位置FP32相加|0|在调制/残差消费warp内计算|0|N/A（准备/已融合/缓存/无对应数学工作）|
|LN1|一行3072维的均值/方差/归一化|224|W3–W6归约/计算；全部8warp搬运|1|0.87× / 0.87× / 0.87×|
|调制1|2048元素，每线程2个四元素向量|336|W0–W7向量计算|1|1.52× / 1.53× / 1.53×|
|Self Q/K/V联合投影|每份输出64×128，K3072；三套权重/输出独立；各两段K1536|288（每投影96）|W0领任务后参与MMA；全部8warp cp.async/MMA/写回，每warp32×32|2|0.86× / 0.86× / 0.86×|
|Self Q RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.79× / 1.79× / 1.79×|
|Self K RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.94× / 1.93× / 1.93×|
|private KV插入|B14只读公共历史，候选private suffix保持独立|0|无|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Self attention|每CTA两组；每组FA2 Q64×KV64×D128，16/32有效Q；KV循环短/中/长5/18/34轮|96/168/96|W0–W3/W4–W7两个独立FA2组，QK/PV Tensor Core、softmax CUDA core|1|4.61× / 4.55× / 9.27×|
|Self输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务后参与MMA；全部8warp cp.async/MMA及写回，每warp32×32|2|0.71× / 0.78× / 0.80×|
|残差1|2048元素，gate按需计算|336|全部8warp向量计算|1|1.83× / 1.86× / 1.86×|
|LN2|一行3072维LN及原FP32 affine|224|W3–W6归约/计算；全部8warp搬运|1|0.79× / 0.83× / 0.80×|
|Cross Q投影|64×128输出，完整K3072，两段K1536|96|W0领任务后参与MMA；全部8warp cp.async/MMA及写回，每warp32×32|2|0.80× / 0.82× / 0.81×|
|Cross K投影|warm直接读取最终K缓存；cold本层两条件各512行，192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross V投影|warm直接读取V缓存；cold192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross Q RMS|两行×3072维RMS，无RoPE|112|两组4warp各算一行|1|0.69× / 0.69× / 0.69×|
|Cross K RMS|最终K已缓存；cold512个两行tile|0|warm无；cold两组4warp|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross attention|每CTA两组；每组Q16×K512×D128，8轮KV64|168|两个独立4warp FA2组，QK/PV Tensor Core、softmax CUDA core|1|1.21× / 1.20× / 1.21×|
|Cross输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务后参与MMA；全部8warp cp.async/MMA及写回，每warp32×32|2|0.74× / 0.74× / 0.74×|
|残差2|2048元素，两BF16输入相加|336|全部8warp向量计算|1|0.64× / 0.64× / 0.73×|
|LN3|一行3072维LN，输出FP32|224|W3–W6归约/计算；全部8warp搬运|1|0.62× / 0.62× / 0.61×|
|调制3|2048元素，按需scale/shift|336|全部8warp向量计算|1|1.54× / 1.54× / 1.54×|
|FFN1|108主块128×256/K32×96；12尾块32×256/K64×48；总K3072|120|主块8warp MMA；尾块W0–W3 MMA；8warp写回|0|与FFN四段合计计时，见下|
|GELU|融合在FFN1写回，不额外启动/领取tile|0|全部8warp epilogue CUDA core|0|N/A（准备/已融合/缓存/无对应数学工作）|
|FFN2|128×128部分和，K7168/64=112轮；每个最终输出两部分，共48输出tile|96|W3–W6 MMA；全部8warp四元素写回；组就绪后领取|0|与FFN四段合计计时，见下|
|残差3 / FFN出口|融合在第二部分FFN2写回；四组生产28/28/28/36，消费各24；仅最终grid同步|0|8warp epilogue；CTA leader发布/领取/有序提交|1|与FFN四段合计计时，见下|
|最终退出|当前B14没有真实cache写回/恢复工作；保留退出屏障|—|108 CTA全体|1|N/A（准备/已融合/缓存/无对应数学工作）|

FFN1→GELU→FFN2→残差3共同区间：**1.02× / 1.01× / 1.02×**。四组生产28/28/28/36，消费各24个部分任务；后半段可提前计算，合并提交等待前半段。不能拆成四项可相加耗时。

当前六个M224投影都使用64×128输出tile、warp32×32、8warp MMA；每投影原两段K1536/各24轮K64及BF16/bias次序不变。QKV联合队列只是让独立投影任务可同时领取，保留三份权重与输出；不是全层ready调度。其他阶段仍有同步，CTA内warp职责随阶段变化。

warm Cross K/V/RMS读缓存；cold另有K投影192、V投影192、K RMS512，共896tile与5次grid同步，不混入本表warm比较。Self attention短/长有允许的双候选共享prefix分组，中档回退单候选；每候选仍Q16，private KV互不可见。Torch的history gather/expand/cat等开销也计入对应阶段。

[原始阶段汇总](CURRENT-STAGE-SUMMARY.json)、[完整层四臂同轮](SUMMARY.json)、[结果和负结果](RESULT.md)、[资源与交接](RESOURCE-AND-STOP-REPORT.json)。当前源码：`/rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/whole_layer.cu`中的`self_qkv_projection_group`、`self_qkv_projection8`和`linear`；对应tile在同目录`cutlass_gemm.cuh`。baseline、投机生产、原tile未改。
