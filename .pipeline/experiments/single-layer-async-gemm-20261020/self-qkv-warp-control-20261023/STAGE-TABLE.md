# 当前完整单层阶段表：独立8warp Self QKV

更新时间：2026-10-23T01:21:17.786298+08:00。源码 `ce9d30a488d7`，完整身份见[SELECTED](SELECTED.json)。两机已采用。B14×Q16×D3072，H24×head_dim128，FFN14336；108常驻CTA，每CTA256线程/8warp，112384B shared、255寄存器/线程、静态local1016B、1CTA/SM。数学tile总数3408/3480/3408，grid同步27，均未改变。

**加速比=原Torch时间/自有时间，顺序为短288/中1152/长2160。大于1自有更快。完整层无插桩warm Graph：1.2661× / 1.8074× / 2.9322×。** 双方均预计算最终Cross K（含RMS）/V，Torch使用静态dense history；不是30层、chunk、机器人E2E，也不是最强guarded增量。

当前版本全部阶段已重新测量，两进程同真实输入。表中自有为完整kernel中CTA0进入/退出区间，包含原队列和同步；Torch为原模块表达式在对应真实中间输入上的独立热Graph。两者缓存上下文不同，阶段比值仅用于定位，不可相加当完整层分解或可省墙钟。测量副本资源与选定版本一致，完整层扰动绝对值≤0.454%；记录的共同中间值、完整输出、输入/有效KV/RNG有限逐位通过。约1.024μs时间戳量化限制微小差异解释。

G为新增grid同步次数，不包含CTA/warp组内部barrier。tile为逻辑输出工作包，不等于独立kernel、常驻CTA或硬件计数。

|阶段|做什么 / 每tile工作量|tile数（短/中/长）|CTA内warp分工|G|原Torch/自有（短/中/长）|
|---|---|---|---|---:|---|
|可见历史layout|CTA0按每轮256物理slot压紧有效历史；实际pool由输入决定|—|仅CTA0的W0–W7|1|N/A（准备/已融合/缓存/无对应数学工作）|
|调制参数表|不再生成m×6×D表，在消费位置FP32相加|0|在调制/残差消费warp内计算|0|N/A（准备/已融合/缓存/无对应数学工作）|
|LN1|一行3072维的均值/方差/归一化|224|W3–W6归约/计算；全部8warp搬运|1|0.87× / 0.87× / 0.87×|
|调制1|2048元素，每线程2个四元素向量|336|W0–W7向量计算|1|1.52× / 1.53× / 1.53×|
|Self Q投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W0–W7均cp.async/MMA；全部8warp写回；每warp32×32|2|0.77× / 0.77× / 0.77×|
|Self K投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W0–W7均cp.async/MMA；全部8warp写回；每warp32×32|2|0.83× / 0.84× / 0.84×|
|Self V投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W0–W7均cp.async/MMA；全部8warp写回；每warp32×32|2|0.82× / 0.85× / 0.85×|
|Self Q RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.79× / 1.79× / 1.78×|
|Self K RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.80× / 1.93× / 1.92×|
|private KV插入|B14只读公共历史，候选private suffix保持独立|0|无|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Self attention|每CTA两组；每组FA2 Q64×KV64×D128，16/32有效Q；KV循环短/中/长5/18/34轮|96/168/96|W0–W3/W4–W7两个独立FA2组，QK/PV Tensor Core、softmax CUDA core|1|4.61× / 4.53× / 9.25×|
|Self输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.65× / 0.70× / 0.70×|
|残差1|2048元素，gate按需计算|336|全部8warp向量计算|1|1.83× / 1.85× / 1.86×|
|LN2|一行3072维LN及原FP32 affine|224|W3–W6归约/计算；全部8warp搬运|1|0.79× / 0.79× / 0.80×|
|Cross Q投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.72× / 0.72× / 0.72×|
|Cross K投影|warm直接读取最终K缓存；cold本层两条件各512行，192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross V投影|warm直接读取V缓存；cold192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross Q RMS|两行×3072维RMS，无RoPE|112|两组4warp各算一行|1|0.68× / 0.68× / 0.68×|
|Cross K RMS|最终K已缓存；cold512个两行tile|0|warm无；cold两组4warp|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross attention|每CTA两组；每组Q16×K512×D128，8轮KV64|168|两个独立4warp FA2组，QK/PV Tensor Core、softmax CUDA core|1|1.19× / 1.19× / 1.19×|
|Cross输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.68× / 0.69× / 0.69×|
|残差2|2048元素，两BF16输入相加|336|全部8warp向量计算|1|0.64× / 0.65× / 0.65×|
|LN3|一行3072维LN，输出FP32|224|W3–W6归约/计算；全部8warp搬运|1|0.60× / 0.60× / 0.60×|
|调制3|2048元素，按需scale/shift|336|全部8warp向量计算|1|1.54× / 1.54× / 1.54×|
|FFN1|108主块128×256/K32×96；12尾块32×256/K64×48；总K3072|120|主块8warp MMA；尾块W0–W3 MMA；8warp写回|0|与FFN四段合计计时，见下|
|GELU|融合在FFN1写回，不额外启动/领取tile|0|全部8warp epilogue CUDA core|0|N/A（准备/已融合/缓存/无对应数学工作）|
|FFN2|128×128部分和，K7168/64=112轮；每个最终输出两部分，共48输出tile|96|W3–W6 MMA；全部8warp四元素写回；组就绪后领取|0|与FFN四段合计计时，见下|
|残差3 / FFN出口|融合在第二部分FFN2写回；四组生产28/28/28/36，消费各24；仅最终grid同步|0|8warp epilogue；CTA leader发布/领取/有序提交|1|与FFN四段合计计时，见下|
|最终退出|当前B14没有真实cache写回/恢复工作；保留退出屏障|—|108 CTA全体|1|N/A（准备/已融合/缓存/无对应数学工作）|

FFN1→GELU→FFN2→残差3共同流水区间：**1.02× / 1.02× / 1.02×**。四组FFN1生产28/28/28/36、每组24个FFN2部分任务；部分2可先计算，合并提交等部分1。不能拆出四项可相加耗时。完整ACT仍保留。

warm Cross K/V/RMS跳过；cold本层两条件各512行，另需192+192+512=896tile与5次grid同步，未混入warm比值。Self attention短/长用允许的双候选prefix分组，中档回退单候选；每候选仍Q16，private KV互不可见。Torch阶段含history gather/expand/cat，比值不等于纯attention MMA加速。

当前Self Q/K/V仍未追平原Torch，但比4warp控制的对应阶段约快10%–15%。Self输出、Cross Q/out仍为4warp，下一步优先核对同M/N/K投影是否适用本次轻量8warp函数；尚未执行，不保证收益。暂不扩大CTA tile或恢复全层ready调度。

[本轮结论](RESULT.md)、[阶段数值](CURRENT-STAGE-SUMMARY.json)、[两进程阶段1](final-stages/RESULT-1.json)、[阶段2](final-stages/RESULT-2.json)、[同轮完整层](SUMMARY.json)。代码：`/rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/whole_layer.cu` 的 `self_qkv_projection8` 与 `linear`；tile定义及计算在同目录 `cutlass_gemm.cuh` 的 `CutlassSelfQKV8` / `self_qkv8_tile`。baseline、投机生产和旧tile未改。
