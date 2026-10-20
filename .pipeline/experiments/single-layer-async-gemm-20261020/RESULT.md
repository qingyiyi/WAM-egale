# M39：单层tile GEMM双缓冲实测

更新时间：2026-10-20T10:45:33.123435+08:00。计算终态 `COMPLETE_ASYNC_GEMM_RESEARCH_UNADMITTED`；原Torch精度和生产准入仍为NO-GO。**另有资源登记字段不一致，资源审计FAILED**，见[资源报告](RESOURCE-AND-STOP-REPORT.md)，不改原始登记或脚本终态。

## OBSERVED：实现与验证

在optimized中实现SM80 cp.async双缓冲，仍为完整一层、一个cooperative persistent kernel。warp0领取CTA任务、1/2加载A/W、3–6执行WMMA、7写回；64×64×32输出/归约分块及K递增累加、BF16 epilogue保持。双buffer与LD40共享布局使下一K块搬运可与当前MMA重叠；未改attention、norm、RoPE、cache、scheduler或stage barrier。

[算法](../../../算子研究/wam_ops/optimized/csrc/gemm_pipeline.cuh)、[封装](../../../算子研究/wam_ops/optimized/whole_layer_pipeline.py)、[构建](../../../算子研究/wam_ops/optimized/build_pipeline.py)、[源manifest](SOURCE-MANIFEST.json)。sync为同LD40同步对照，async为新双缓冲，profile只是async的计时副本；不是三种生产候选。用户随后明确后续直接改optimized、不为带读保留额外版本、不另建登记；本轮旧源码身份及测量关系仍以源manifest保存。

共9真实block0：action B2/B14、Q16，短/中/长有效历史288/1152/2160；video B2、Q128，历史272/1136/2144。D3072、H24、head_dim128、FFN14336、cross512。B14仍14个独立Q16注意力实例，不是Q112；私有KV语义保持。

- native对原Torch最终输出9/9逐位通过；新async对原Torch最终输出9/9逐位失败，继承旧原型差异，不能写“精度已对齐”。
- sync/async/profile相对旧whole在9点全部保存中间量、输出、metadata及全部逻辑有效KV相同；无效KV不比较。输入、RNG、参数身份、tile exactly-once及新async重复确定性通过。
- 6个B2 action/video起点，合法update1→2→0，共18个转移，旧whole和新async输出及有效状态相同。
- 两独立计时进程各9个Chrome trace均恰好1个compute kernel，共18份；memcpy/memset/annotation不计作计算kernel。
- 这些只是有限真实fixture验证；不代表所有输入，不包含接受路径、完整30层、chunk或机器人闭环。

## OBSERVED：完整单层时间

远端A100-SXM4-40GB GPU0，无观察到其他GPU context；CPU4/5两核。Torch2.9.0+cu126，扩展CUDA12.4/SM80。两个独立进程，每cell每进程6次交错CUDA-event计时，432个样本全部保留。下表每cell为12样本的合并中位数，单位ms；profiler和状态准备在计时外。相对改进只比较本轮同输入、同一时段旧whole；native为成熟库完整block，与原Torch有限逐位一致。

| 输入 | native | 旧whole | 同布局sync | 新async | 相对旧whole耗时下降 | 新async/native耗时 |
|---|---:|---:|---:|---:|---:|---:|
| B2 action / 短 | 1.480 | 7.567 | 6.672 | 4.932 | 34.8% | 3.33× |
| B2 action / 中 | 1.629 | 8.637 | 7.740 | 5.999 | 30.5% | 3.68× |
| B2 action / 长 | 1.758 | 9.737 | 8.871 | 7.150 | 26.6% | 4.07× |
| B14 action / 短 | 2.456 | 27.278 | 26.171 | 17.371 | 36.3% | 7.07× |
| B14 action / 中 | 2.699 | 29.250 | 26.588 | 19.369 | 33.8% | 7.18× |
| B14 action / 长 | 3.070 | 32.535 | 29.881 | 22.909 | 29.6% | 7.46× |
| B2 video / 短 | 1.929 | 13.576 | 12.167 | 9.781 | 28.0% | 5.07× |
| B2 video / 中 | 2.046 | 17.001 | 15.580 | 13.227 | 22.2% | 6.47× |
| B2 video / 长 | 2.182 | 20.814 | 19.429 | 17.098 | 17.9% | 7.84× |

新完整block相对旧原型耗时下降17.9%–36.3%，加速1.22–1.57×；与同LD40同步对照比加速1.14–1.51×，说明收益不全来自padding。仍比native慢3.33–7.84倍，不能称为超越Torch的加速。合并中位数的比值不必等于各进程比值的平均；原样本与逐配对比值全部在[RESULT.json](RESULT.json)/[TIMINGS.csv](TIMINGS.csv)，不选最好轮次。

| 输入 | 进程1旧whole/async | 进程2旧whole/async | 合并sync/async |
|---|---:|---:|---:|
| B2 action / 短 | 1.535× | 1.533× | 1.353× |
| B2 action / 中 | 1.441× | 1.437× | 1.290× |
| B2 action / 长 | 1.364× | 1.357× | 1.241× |
| B14 action / 短 | 1.601× | 1.593× | 1.507× |
| B14 action / 中 | 1.510× | 1.510× | 1.373× |
| B14 action / 长 | 1.420× | 1.424× | 1.304× |
| B2 video / 短 | 1.389× | 1.383× | 1.244× |
| B2 video / 中 | 1.278× | 1.285× | 1.178× |
| B2 video / 长 | 1.217× | 1.228× | 1.136× |

两进程各cell中位数都改善；所有本轮逐repeat旧whole/async比也大于1。没有置信区间或普适性能保证。此处是未准入原型的机制测量，uncached eager对照，不是最强cached/Graph/shared-prefix路径，不是机器人E2E。

## OBSERVED：阶段归因

独立globaltimer插桩副本，每case每路径3次。GEMM十阶段总时间改善1.73–1.95×。下表比例的分母是新插桩副本kernel时间；不可与上方未插桩时间直接相减。

| 输入 | 新GEMM总占比 | 新self-attention占比 | 新cross-K/V投影占比 | GEMM旧/新 |
|---|---:|---:|---:|---:|
| B2 action / 短 | 68.5% | 7.4% | 32.9% | 1.848× |
| B2 action / 中 | 55.6% | 22.7% | 26.6% | 1.845× |
| B2 action / 长 | 45.6% | 35.2% | 22.0% | 1.834× |
| B14 action / 短 | 73.4% | 6.1% | 52.4% | 1.950× |
| B14 action / 中 | 62.1% | 20.5% | 44.2% | 1.950× |
| B14 action / 长 | 50.1% | 36.0% | 35.6% | 1.946× |
| B2 video / 短 | 53.3% | 10.7% | 15.7% | 1.732× |
| B2 video / 中 | 38.6% | 34.5% | 11.4% | 1.757× |
| B2 video / 长 | 29.3% | 49.4% | 8.7% | 1.758× |

新async172 registers/thread、59616B shared、local bytes0，理论1 CTA/SM，108CTA×256threads；sync174、profile180 registers。寄存器与shared并未出现足以解释该提速的驻留变化；理论值不是achieved occupancy。K3072内部CTA barrier194→99、K14336的898→451是源码静态计数，并非实测stall比例。

## HYPOTHESIS / UNKNOWN

双缓冲、cp.async缓存提示和较少CTA barrier共同组成这次改动；sync控制了LD40布局，但没有分别隔离这三个因素，不能把所有改进归因于纯加载重叠。NCU硬件计数仍UNKNOWN；L2/DRAM流量、stall和实际occupancy未测。可选SASS核查因现有工具链没有cuobjdump而未执行，不虚报LDGSTS数量；保留CPU失败记录。

新B14短KV仍由cross K/V投影主导（52.4%）；长KV action的self-attention约35%–36%，video长KV49.4%。下一优化应分别处理GEMM效率与attention数据通路；本轮没有证明更大tile或ready队列能加速。

## RECOMMENDATION：直接在optimized继续

先评估GEMM K32→K64，保持输出M64/N64及原16维MMA次序，用更少循环与同步换额外shared；不盲目放大所有维度。B2 action实际M32，扩大M会增加补零；cross-K/V M1024/7168可再考虑N128，但需要检查寄存器和任务并行度。Attention action保持Q16边界，先处理KV加载与softmax；任何改变归约树的做法重新核对数值。

这些是下一步建议，未运行，未另建登记；沿当前optimized与现有记录继续。baseline、投机源、原tile、旧FAILED和四项false保持。结果不能给生产路径或H3/H4签证。
