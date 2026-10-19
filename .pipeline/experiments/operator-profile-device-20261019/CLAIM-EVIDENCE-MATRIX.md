# 本轮claim与证据

|claim|观察证据|缺口/限制|允许写入|
|---|---|---|---|
|实际action GEMM M32→224|六真实block的aten::addmm与M/N/K|只有1/7状态，无3状态与全部sigma|是，限定本fixture|
|每候选仍独立Q16|真实输入与shared/private实现|不是跨候选Q112序列|是|
|cached/shared/Graph组合保持输出/状态|30eager+24Graph配置逐位过|静态admit；仅block0/update0；非生产失效协议|是，限定范围|
|FFN为有意义连续子图|强路径trace kernel占21.9–29.9%|不是完整WAM时间占比；profiler非正式计时|是，限定范围|
|完整device FFN等价|FFN1/GELU过，FFN2六点失败|差异根因未确认，四种分块公式均未复现|否，明确NO-GO|
|FFN1+GELU保持中间与完整block结果|六真实fixture×两个进程，中间/输出/KV/metadata/RNG/参数逐位过|非所有输入数学定理；未测新30层/policy|是，有限通过|
|FFN1+GELU超过强block基线|两个独立进程15轮交错；B14延迟下降2.451–3.232%|NVML异常、绝对延迟漂移、无正式独占证书；通用融合|仅探索性block数值|
|显著HBM/L2减半|NCU原始counter|默认缓存重放与时钟控制偏差；sectors≠requests；无新融合counter|否|
|H2同7-query批处理经济性|仅形状与两种M有限profile|尚未同冻结7-query逐个/集中对照|未完成|
|H3候选感知调度/协同|通用FFN小增量|无P_seq/P_sched、无候选DAG调度|否|
|H4有效提交和闭环加速|本轮block-only|g/round/chunk/episode未测，四质量权限false|否|

全部历史FAILED与四项false保持。component/local门与工程质量、独立TEST、非劣和正式性能权限分开。
