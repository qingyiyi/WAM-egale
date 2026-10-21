# 当前08逐阶段控制（非正式性能）

OBSERVED：两个独立进程×3KV×28计算阶段，输出/共有中间量/有效KV/输入/RNG有限逐位通过。每个独立stage使用相同原版前缀生成的真实输入。原样one_stage共享配额100096B、256线程；整层同控制流插桩，比活动08约增加0.3%–0.4%。阶段时间包含原stage内部领取、barrier及device调用，不是纯算术。外部marker含kernel边界，不能把两者相减精确称为全局同步耗时。

|中KV阶段|原版同前缀 µs|自有独立/整层资源 µs|自有整层嵌入 µs|
|---|---:|---:|---:|
|table|30.720|32.768|29.696|
|self_Q|43.008|55.808|55.296|
|self_K|38.912|53.248|45.056|
|self_attention（原版列为强guarded）|156.672|237.568|232.448|
|cross_K|553.984|580.608|574.464|
|cross_V|552.960|580.608|569.856|
|cross_K_RMS|69.632|118.784|117.760|
|cross_attention|99.328|122.880|116.736|
|FFN1|166.912|216.064|212.992|
|GELU|15.360|31.744|35.840|
|FFN2|116.736|147.456|140.800|

全三KV/两进程见PHASES.csv，原始RESULT-1/2.json保留。原版self_attention另有dense pack和sharedTorch，不以慢dense替代强guarded。

HYPOTHESIS：shared/register固定配额及资源约束、device函数组织、每阶段工作几何均可能限制整层；不是“全部因为grid.sync”。独立stage沿用整层shared配额，不能据此宣称原语在自由资源下也慢同样倍数。CrossK RMS独立寄存器62/localSize0仍约119us，整层约118us；这个对照不支持把其差距主要归因于全层特有的寄存器spill或跨阶段等待。其shared依然100096B、仅1CTA/SM，尚需独立资源控制验证。

UNKNOWN：本轮NCU动态cache/local/stall细项尚未取得；local-ncu-08在外部GPU任务进入后启动门拒绝，0 GPU生命周期。静态localSize仅是地址空间/栈资源，不能冒充实际spill流量。新geometry是否消除差距待下一项对照。

RECOMMENDATION：小投影按实测原版M128×N128×两K1536做同数学对照；FFN1最多一次N256低shared对照。先对齐原语及资源，不启动DAG。
