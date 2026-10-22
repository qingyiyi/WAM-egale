# 当前完整单层阶段表：outer-inline + FFN1 tail K64

对应当前选定源码 `9b5dec45…`；完整身份见 [SELECTED.json](SELECTED.json)。本表覆盖 B14 action：B14×Q16×D3072，24 heads、head_dim128、FFN14336，短/中/长KV288/1152/2160。一次persistent cooperative kernel启动108 CTA，每CTA256线程/8warp；后续反复处理tile，不是启动3408个CTA。B2原action仍使用原路径，不把本表外推到B2。

两臂均预计算最终Cross K（含RMS）/V，冷构建单列。自有每CTA shared112384B，255寄存器/线程，静态local1016B/线程，当前1CTA/SM。warp职责随阶段变化，并非专用scheduler/GEMM/epilogue warp全程异步工作。

**加速比统一为原Torch耗时÷自有耗时；大于1自有更快，小于1自有更慢。** 三个数字始终为短/中/长KV。

无插桩完整层、两独立进程同输入交错Graph实测：**1.2426× / 1.7780× / 2.8859×**。这是完整单层warm成本，不是30层、chunk或机器人E2E，也不是相对历史最强guarded引擎的新收益。

## 阶段测量边界

下表阶段比值属于诊断：自有为当前完整kernel内CTA0的进入/退出时间，包含现有调度和阶段同步；Torch用原模块/原表达式在同一真实中间输入上独立Graph热回放。不是同一缓存上下文的独立原语比赛，不能逐项相加当整层分解或因果可省时间。调制表Torch单独物化约47μs；自有按需计算已计入对应消费者，不报无穷加速。

测量副本不替换生产DSO。第一版因Params布局变化使静态local增至1056B、完整层扰动1.3%–1.7%，全部保留。第二版复用原有trace指针，两进程的寄存器/shared/local/驻留资源与选定版本一致；normal/off/on完整层扰动绝对值最大0.461%。完整输出、共同中间值、每个Torch表达式输出、输入/有效KV/RNG有限逐位核验通过。微秒级时间戳有量化误差；接近1的小差异不据此宣称稳定原语优势。

## 从上到下的计算与分块

G表示此行新增的grid同步次数；tile内CTA/warp组同步及FFN发布/领取开销不在G内。下列tile为逻辑输出工作包，不是GPU硬件counter测出的CTA或MMA指令数。

|阶段|做什么 / 每tile工作量|tile数（短/中/长）|CTA内warp分工|G|原Torch/自有（短/中/长）|
|---|---|---|---|---:|---|
|可见历史layout|CTA0按每轮256物理slot压紧有效历史；实际pool由输入决定|—|仅CTA0的W0–W7|1|N/A（准备/已融合/缓存/无对应数学工作）|
|调制参数表|不再生成m×6×D表，在消费位置FP32相加|0|在调制/残差消费warp内计算|0|N/A（准备/已融合/缓存/无对应数学工作）|
|LN1|一行3072维的均值/方差/归一化|224|W3–W6归约/计算；全部8warp搬运|1|0.86× / 0.87× / 0.87×|
|调制1|2048元素，每线程2个四元素向量|336|W0–W7向量计算|1|1.51× / 1.53× / 1.52×|
|Self Q投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.68× / 0.68× / 0.68×|
|Self K投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.72× / 0.72× / 0.72×|
|Self V投影|64×128输出，完整K3072；两段K1536，各24轮K64；最后M块32有效行|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.73× / 0.74× / 0.74×|
|Self Q RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.72× / 1.73× / 1.78×|
|Self K RMS+RoPE|两行×3072维RMS，再旋转3072复数对；RMS后先BF16舍入|112|W0–W3/W4–W7各算一行，归约及旋转CUDA core|1|1.80× / 1.79× / 1.79×|
|private KV插入|B14只读公共历史，候选private suffix保持独立|0|无|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Self attention|每CTA两组；每组FA2 Q64×KV64×D128，16/32有效Q；KV循环短/中/长5/18/34轮|96/168/96|W0–W3/W4–W7两个独立FA2组，QK/PV Tensor Core、softmax CUDA core|1|4.60× / 4.52× / 9.27×|
|Self输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.68× / 0.72× / 0.71×|
|残差1|2048元素，gate按需计算|336|全部8warp向量计算|1|1.83× / 1.90× / 1.95×|
|LN2|一行3072维LN及原FP32 affine|224|W3–W6归约/计算；全部8warp搬运|1|0.79× / 0.79× / 0.80×|
|Cross Q投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.72× / 0.72× / 0.72×|
|Cross K投影|warm直接读取最终K缓存；cold本层两条件各512行，192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross V投影|warm直接读取V缓存；cold192个128×128输出tile|0|warm无；cold W3–W6 MMA、全部warp写回|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross Q RMS|两行×3072维RMS，无RoPE|112|两组4warp各算一行|1|0.68× / 0.68× / 0.68×|
|Cross K RMS|最终K已缓存；cold512个两行tile|0|warm无；cold两组4warp|0|N/A（准备/已融合/缓存/无对应数学工作）|
|Cross attention|每CTA两组；每组Q16×K512×D128，8轮KV64|168|两个独立4warp FA2组，QK/PV Tensor Core、softmax CUDA core|1|1.20× / 1.20× / 1.19×|
|Cross输出投影|64×128输出，完整K3072，两段K1536|96|W0领任务；W3–W6 MMA；全部8warp写回|2|0.69× / 0.69× / 0.69×|
|残差2|2048元素，两BF16输入相加|336|全部8warp向量计算|1|0.64× / 0.64× / 0.64×|
|LN3|一行3072维LN，输出FP32|224|W3–W6归约/计算；全部8warp搬运|1|0.60× / 0.59× / 0.60×|
|调制3|2048元素，按需scale/shift|336|全部8warp向量计算|1|1.53× / 1.53× / 1.47×|
|FFN1|108主块128×256/K32×96；12尾块32×256/K64×48；总K3072|120|主块8warp MMA；尾块W0–W3 MMA；8warp写回|0|与FFN四段合计计时，见下|
|GELU|融合在FFN1写回，不额外启动/领取tile|0|全部8warp epilogue CUDA core|0|N/A（准备/已融合/缓存/无对应数学工作）|
|FFN2|128×128部分和，K7168/64=112轮；每个最终输出两部分，共48输出tile|96|W3–W6 MMA；全部8warp四元素写回；组就绪后领取|0|与FFN四段合计计时，见下|
|残差3 / FFN出口|融合在第二部分FFN2写回；四组生产28/28/28/36，消费各24；仅最终grid同步|0|8warp epilogue；CTA leader发布/领取/有序提交|1|与FFN四段合计计时，见下|
|最终退出|当前B14没有真实cache写回/恢复工作；保留退出屏障|—|108 CTA全体|1|N/A（准备/已融合/缓存/无对应数学工作）|

FFN1→GELU→FFN2→残差3共享流水区间的诊断比值：**1.01× / 1.01× / 1.03×**。不可再拆成四个独立阶段耗时或重复计入。FFN1的四组生产数28/28/28/36；每组就绪后可派24个FFN2部分任务，第二部分可先计算但合并提交等待第一部分；完整ACT缓冲仍保留。当前有限诊断说明该区间接近Torch，不签“所有FFN形状均追平”的结论。

总数学tile **3408 / 3480 / 3408**，总grid同步 **27**。其中6个小投影各96tile，每次含入口队列初始化与出口两次grid同步；静态norm/point各一次出口。FFN只有最终出口grid同步，内部使用CTA同步、就绪计数、CAS领取和提交依赖。

未缓存Cross路径还需K投影192、V投影192、K RMS512，共额外896tile和5次grid同步；这不是当前warm计时。原TensorCore支持的GEMM、attention QK/PV保持；RMS、RoPE、GELU、softmax、残差在CUDA core执行。

Self attention短/长使用允许的双候选共享prefix分组，中档回退单候选；每候选仍Q16，私有K/V不互相可见。表中32有效query行表示两份独立候选的query工作包，不是一个允许互相注意的新序列。此阶段原Torch包含history gather/expand/cat，因而比值不能单独归功于MMA或warp调度。

## 当前结论与下一步

- 最大明确剩余差距在六个小投影：约0.67–0.74×，即自有约慢35%–49%。其累计额外时间比当前FFN接近相等的差异更值得优先定位。
- LN3约0.59×、LN2约0.79×、Cross Q RMS约0.68×、残差2约0.63×；它们各自规模小，不能只按比例决定优先级。
- Self attention约4.5–9.3×，Cross attention约1.2×；含共享历史组织、减少物化等既有收益，不能冒充全是本轮FFN优化。
- 后续优先小投影同MMA体的函数边界、动态领取/两处grid及写回的有限分离；当前数据尚不能把差距归因成单一全局同步。先不恢复全层ready队列，也不扩大tile网格搜索。

[阶段数据汇总](CURRENT-STAGE-SUMMARY.json)、[源码数学映射](STAGE-SOURCE-MAP.json)、[两进程原始软件时间线](current-stage-profile/r2-existing-trace/RESULT.json)、[第二进程](current-stage-profile/r2-existing-trace/RESULT-2.json)、[无插桩完整层](tail-k64/SUMMARY.json)。

当前源码入口：`/rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/whole_layer.cu:923`；FFN流水从427行、FFN1尾块315行、小投影157行；GEMM tile定义在同目录 `cutlass_gemm.cuh`，FA2阶段在 `fa2_stage.cuh`。baseline和投机代码未修改。
