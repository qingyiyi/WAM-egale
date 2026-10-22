# 单层对齐晨间结果（2026-10-22T07:57:53.404180+08:00）

**三项新对照均未晋级；保留昨晚selected11，完整层尚未完全对齐同工作量guarded强基线。** 本机/BAAI源与活动DSO已经核对恢复，源0cc1944b、DSOac3f9e46。优化仍只有一层、一个persistent计算kernel，未实施跨阶段ready-DAG。

## OBSERVED：现用版本与本轮对照

|B14 KV|昨晚开始ms|现用版本ms|同轮uncached guarded ms|现用仍慢|
|---|---:|---:|---:|---:|
|288|2.510|2.296|2.070|10.94%|
|1152|2.718|2.453|2.187|12.17%|
|2160|2.977|2.476|2.201|12.50%|

上表是[原两进程完整层表](../overnight-alignment-20261022/WHOLE-LAYER-COMPARISON.csv)，不是新机器人E2E。本次同卡重复旧版约2.292–2.295/2.459–2.461/2.473–2.477ms，与旧证据相近，不挑最快值覆盖旧表。所有性能来自完整层Graph、同输入、两独立进程；NCU时延不进入表。cache projection=false，不能声称已经对齐全部cached引擎。

|晨间单因素|短KV耗时变化|中KV|长KV|决定|
|---|---:|---:|---:|---|
|B14内层FA2 forceinline|+0.112%|−0.052%|+0.103%|无稳定收益，撤回|
|self/cross路由模板专化|+0.112%|+0.323%|+0.196%|无收益，撤回|
|补齐compile/device-link LTO|+15.564%|+16.146%|+15.669%|稳定退化，撤回|

变化是两进程各24次中位数均值的比，原逐进程结果与source/checkpoint身份见[THREE-TRIALS.csv](THREE-TRIALS.csv)、[汇总JSON](THREE-TRIALS.json)。每项先完成9个真实签名、81完整调用，原Torch输出/23共有中间量/有效KV/普通debug/输入/RNG有限逐位通过，再运行计时；三项通过精度不等于通过性能。现用11源和DSO未变，原21结构变体84调用证据继续保留，本次没有重复运行它们。

原晨间上限2项在实际编译链新证据出现后前瞻增到3项，[修订原文](PLAN-AMENDMENT-1.md)保留，不伪称原2项内。LTO最初与Torch显式-gencode冲突，CPU1.666秒退出、0GPU，03旧FAILED保留；03b用文档规定lto_80完成同一候选。

## OBSERVED：当前版本NCU

[完整4范围计数](local-ncu-reference-fixed/RESULT.md)、[原CSV](local-ncu-reference-fixed/NCU.csv)、[口径说明](COUNTER-INTERPRETATION.md)。真实中KV/B14、当前DSO：active warp12.50%，Tensor活跃47.93%，local读写243623424B（232.337MiB），DRAM1422800128B，barrier warp stall40.67%。原生SelfQ/FFN1/FFN2各范围local0B。

这些数不能解释为232MiB都进入HBM，或40.67%墙钟等待grid.sync。整层含非Tensor计算，也不能用它的47.93%直接与单个GEMM的80%相除。准入无计算/图形PID、显存0，NVML残留24–25%，未改clock/cache，计数只作诊断；NCU实际版本2026.1.1，虽然路径名带cuda12.4。

本机一次因reference扩展旧ABI失败，另一次分阶段NCU在他人sglang进程进入前停止；这些旧终态保留。隔离加载与远端相同且已验证的reference DSO后完成4范围核验，随后修正本机optimized依赖；baseline代码/DSO未改。见[依赖收据](dependency-install/RECEIPT.json)。

## OBSERVED：LTO为什么不保留

三KV×28阶段两进程诊断全部完成，[完整表](lto-stage-diagnostic/RESULT.md)、[逐阶段变化](LTO-STAGE-CHANGE.csv)。插桩整层比普通LTO候选多0.232%–0.664%，普通整层退化约16%独立成立。中KV例：

|阶段|原11整层内µs|LTO整层内µs|LTO独立阶段µs|
|---|---:|---:|---:|
|Cross-K投影|574.208|772.096|565.248|
|Cross-V投影|569.856|768.000|566.016|
|self-attention|230.912|270.336|215.552|
|Cross-K RMS|116.480|141.312|116.992|
|FFN1|198.912|178.944|181.504|
|FFN2|138.240|124.928|125.952|

主要正增量落在Cross-K/V嵌入整层；独立相同计算并不慢。与此同时FFN1/FFN2改善，不能一概宣称LTO对所有原语无效。这里比较两批同协议阶段诊断，native前缀/输入、频率与资源条件有记录，不是同轮逐阶段旧/新交错；阶段中位数相加或相减均不能精确当作全局同步成本。

LTO前后整层仍255regs/thread、100096B动态shared、108CTA；静态local936→816B。LTO的独立Cross-K/V阶段254regs、静态local16B，整层插桩则255regs、800B。静态SASS/源行映射显示local位置从函数边界转向更多计算相关位置；符号合并也没有消除所有内部CALL。[静态证据](STATIC-LOCAL-SOURCE-LOCATIONS.json)不是动态热点计数。

## HYPOTHESIS / UNKNOWN

嵌入后的编译上下文、活跃状态和寄存器分配是LTO退化的强嫌疑；目前不能确定具体哪条spill或指令缓存事件贡献多少。统一资源限制已有独立干预证据。当前selected11的232MiB动态local也不能全部归到attention；LTO负候选没有独占NCU，不移用旧counter。

本轮没有online层替换、30层/完整chunk/任务性能、有效提交成本或闭环质量新增结果。工程质量、独立TEST、非劣和TF正式性能四权限仍false。

## RECOMMENDATION与终态

停止全局内联/全局LTO尝试，继续保留11。下一轮先检验能否保留Cross-K/V现有快速编译路径，同时只改善FFN的设备函数边界；选择性设备编译是待验证方案，仍可保持单kernel，不等于调用外部cuBLAS。B14专用入口缩短活跃状态作为后备，不同时展开两套搜索。必须原精度门＋三KV强对照后才讨论ready-DAG。

缓存命中时Cross-K/V投影可能根本不执行，因此不能依靠这些不存在的tile去填108CTA；后续先区分uncached/cached真实DAG。详见[下一步依据](NEXT-ALIGNMENT-DECISION.md)。

本机与远端本轮全部owned进程已退出；本段保守计费531.767秒，M39累计6004.497/36000秒，CPU准备952.151秒另列。[资源/停止报告](RESOURCE-AND-STOP-REPORT.md)。不是预算耗尽；在前瞻有界范围内停止，不将余额自动挪给其他任务。原09:00截止保留。

RoboTwin无新任务，原01:26截止已过；已有工程4局与DEV90记录不变。LIBERO6000局及其暴露/首两S/长尾缺口不变。HW短入口指向本结果，不把整篇追加到所有索引。
