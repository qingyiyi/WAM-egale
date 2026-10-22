# 当前计数支持什么

OBSERVED：冻结ac3f9e46、中KV/B14真实输入，本机4范围NCU完成。整层108CTA×256线程，255regs，NCU总shared101120B（动态workspace100096B，不能混为同一个字段），active warp12.50%，Tensor活跃47.93%，local读130045440B＋写113577984B，共243623424B（232.337MiB），DRAM1422800128B。原生SelfQ/FFN1/FFN2各阶段local0B。原生FFN1/FFN2 Tensor活跃83.71%/80.80%，但是它们只是GEMM，不能与包含Norm/attention/point的整层平均百分比直接当效率倍数比较。

OBSERVED：整层barrier warp stall40.67%、long scoreboard14.51%。这些是issue/warp统计，不是墙钟的40.67%/14.51%；也不能把barrier都叫grid.sync。原SASS有callee保存/恢复与多个嵌套FA2函数；静态指令数不代表执行次数。阶段资源干预和完整层阶段计时来自夜间独立证据。分阶段NCU在07:28 hlj的sglang scheduler PID3264276进入后未启动GPU，保持缺口。

HYPOTHESIS：统一资源和函数调用边界共同造成嵌入成本。第一项只强制内联两个B14 FA2主循环，用完整层同轮对照判断是否收益。即使local下降，也不预设时间必下降。

UNKNOWN：没有当前完整native层的counter，没有无干扰的分阶段counter，也没有kernel-replay原始次数；不能写整层HBM相对native变化、将232MiB都归因attention、或用这次profile延迟算加速。

口径参考：[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)，2026-10-22核查。local含线程私有栈、编译器放置和spill，流经缓存；barrier等待可来自warp工作不均。未改clock/cache，准入没有C/G PID且显存0，但NVML残留24–25%，仅诊断。

进一步静态核验（07:37）：当前B14 dual主循环已经内联在attention_dual；原独立whole_standard/whole_split符号属于其他路由，不应归到B14。第一项forceinline只改变该dispatcher的2条静态LDL/STL位置（119/118→117/116），29个其他函数编码不变，未消除调用。其整层无稳定收益保留。

ELF重定位确认attention_dual有24个shuffle-bfly和2个barrier包装静态调用点；整层入口另有165个shuffle-down、80个barrier静态调用点。没有动态执行次数，不据此推算时间。实际compile缺LTO IR而link有dlto，第三项只补齐构建链，原两个源修改恢复。
