# 10/21晨间：先对齐逐元素和FFN2

固定本文件所在commit审阅；历史paused72与四项false不变。

当前活动源码只保留table/GELU四元素搬运改进：完整B14单层两进程均改善2.7%–3.1%，
相对已对齐native的短/中/长仍约慢42%/44%/30%，尚未整体速度对齐。
九真实签名、23共享中间量、18有效KV状态与输入/RNG/重复有限逐位通过。不是任意输入证明或闭环准入。

真实Torch GELU是vec4 kernel；此前native代理是标量索引并行kernel。真实调制表包含cast+add两个kernel。
剥离后自有vec4 GELU约比Torch慢4%–6%；固定108 CTA的无队列标量版仍明显慢，不能把全部差距归因同步。
FFN2去中间全局写回没有稳定大收益；M128串行更慢，M128两段K并行的独立版本约比Torch慢4%–6%。
但后者放回整层无稳定收益、短KV慢2%–4%，已撤回，原始源/结果保留。当前活动FFN2仍原M64串行分段。
静态local变化不是动态spill根因证明；NCU计数未知，没有新增跨阶段ready调度。

[算子完整报告](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md) /
[所有独立算子数据](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/ISOLATED-STAGES.csv) /
[完整层所有轮次](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/FULL-LAYER.csv) /
[资源与止损](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESOURCE-AND-STOP.md)。

[整体/分阶段绝对耗时与同步归因边界](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/TIME-BREAKDOWN-20261021.md)：当前完整层与vec4前阶段矩阵分列，不能将C−B当纯同步时间。

代码：[whole_layer.cu的Point4/point](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、
[活动GEMM](../算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh)、
[独立FFN2并行诊断](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/ffn2-parallel/isolated_cutlass.cuh)、
[整层未选试验](../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/full-ffn2-parallel/tested-source/csrc/whole_layer.cu)。
后两者不是活动生产入口。

RoboTwin：Target48采集41成功7失败；固定parent18补齐，13成功5自然失败，FIT12/6269行与CAL6/3078行。
两组不是相同初态，不能直接比较成功率。原0动作PARTIAL、补采自然失败和训练解析错误0更新均保留。
修复后CHILD2500/CONT3000各500更新完成，每个权重10771已有CAL query的GPU adapter/decode逐位通过，
tau固定不调；B2/B14 teacher与闭环质量尚未因此通过。CURRENT18、ON3000、DEV90未启动，下一窗口接续。
[RoboTwin晨报](../.pipeline/experiments/robotwin-pilot-20261020/MORNING-RESULT-20261021.md)。两GPU已无本轮计算进程。

网页审阅请求：先解释独立FFN2接近Torch而嵌入整层没有收益的原因。区分静态资源、动态访存、同步和编译影响；
给最小可证伪实验，不能把未测spill、barrier或E2E写成结论。先对齐原语，不跳到跨阶段ready。

四科学权限false；baseline、原tile、LIBERO生产与旧retention/FAILED不变。
公开快照不含权重、原始PT/KV或DSO，不可据此声称独立重算全部真实张量。
