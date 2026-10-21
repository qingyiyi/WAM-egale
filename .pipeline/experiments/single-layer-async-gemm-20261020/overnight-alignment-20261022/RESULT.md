# 夜间单层性能对齐：阶段结果（2026-10-22）

**已有改善，但尚未完全对齐最强 B14 同工作量基线。** 最终选中11的数学实现，注释修正后重建；本机和BAAI的 optimized 已安装同一个验证二进制。12项实现候选全部收口，不再继续无依据结构搜索；当前只剩本机NCU自然空闲窗口。该状态不是所有输入证明，也不是闭环质量放行。

## OBSERVED：完整单层、三种KV

BAAI GPU0 A100-SXM4-40GB，同轮无争用、CUDA Graph，600次预热/臂，24次轮换交错样本，两个独立进程。单位ms；下表为两进程中位数的均值，逐进程见[原始比较表](WHOLE-LAYER-COMPARISON.csv)。不是profiler时间，不含机器人环境/reset，也不是30层verifier或E2E。

|B14 action，含当前16 token的KV长度|今晚开始自有版|最终自有版|Torch静态原数学路径|shared Torch|guarded强对照|相对今晚开始延迟下降|相对guarded仍慢|
|---|---:|---:|---:|---:|---:|---:|---:|
|288|2.510|2.296|2.315|2.105|2.070|8.52%|10.94%|
|1152|2.718|2.453|3.101|2.255|2.187|9.75%|12.17%|
|2160|2.977|2.476|4.336|2.448|2.201|16.81%|12.50%|

以上各臂均uncached context projection、相同完整层工作。Torch列使用既有PreparedBlock提前确定合法slots的静态Graph路径，其投影/norm/RoPE/attention/FFN沿原Torch数学；不是原动态Python入口计时。不能挑较慢dense attention宣布超过所有最佳路径。cached投影矩阵、本轮B2/video正式计时、完整模型/任务计时均未补做。

## 精度、状态、身份

- [9真实签名](final-whole-comparison/SIMPLE-GATE-RESULT.json)：B2 action、B14 action、B2 video各短中长KV，共81次完整调用；原Torch最终输出、23共有中间量、普通/debug、重复、B2 update1→2→0全部有效KV、输入/RNG通过有限逐位比较。
- [21结构变体](final-whole-comparison/RESULT.json)：3KV×原始/drop1/drop16/drop32/holes16/CFG交错/CFG交换，共84次完整调用；完整输出、中间量、合法状态有限逐位一致。有限fixture仅验证序列化的逻辑输入契约，不能证明未保存的原GPU别名/stride或所有在线输入。
- 08扩展short/drop1曾有2个self-attention中间元素相差1.49e-8，最终输出虽相同仍判失败。08b对非16尾走G1后通过；原失败记录保留，不放宽容差。
- 最终源码`whole_layer.cu` SHA256 `0cc1944b69fd10e1aa79c16f79852de7ebd863e63a2a88b827fbb24f9e1c4ae0`；DSO `ac3f9e46574713fc8555428662c4ca40e1eab0c2d2ba830cc4050ccf713a4ad4`。全部相关源清单见[SELECTED.json](SELECTED.json)。[本机部署](DEPLOYMENT-LOCAL.json)、[远端部署](DEPLOYMENT-BAAI.json)保留旧2a8二进制，未触碰baseline/原tile/投机生产。
- 三种B14计时fixture路径与精度fixture相同；每个计时臂也现场检查输出、有效KV、输入和RNG。没有借用另一组输入的精度结果。

## 保留了什么，为什么有效

1. pointwise和norm按CTA静态步进领取合法工作；消除每个小tile的atomic领取及重复屏障。数学、vec4计算、归约和BF16边界不变。
2. GEMM共享FP32写回缓冲stride128→136，使用原向量化写回；保持FFN2分段结果的有序BF16合并。
3. 将现有guarded共享历史self-attention嵌入单层；保留CFG/私有KV边界和非16尾G1。该attention算法本来就有，不能重新申报新算法。
4. B14的self/cross attention在每CTA内使用两组4warp，分别领取合法任务并使用独立workspace/命名barrier；全部8warp参与attention计算。
5. FFN1改为M128×N256×K32、8个MMA warp，三stage主循环；FP32 epilogue按N两半复用128×136 shared区域，避免扩大整个kernel共享内存。单独贡献较小，以完整层改善决定保留。
6. 删除静态阶段无需的队列重置/重复入口grid同步；上阶段出口依赖仍保留，动态队列和真正KV更新的同步未删除。B14无私有cache写入时直接返回。不是ready-DAG，也没有把所有同步移除。

仍是**单个完整层persistent kernel，108CTA，256线程/CTA**。阶段间存在真实依赖；没有把self/cross attention改成7候选互相可见的Q112序列。当前模型hidden3072、FFN14336、24head×128；Q16。小投影M224，条件K/V投影M7168，不能混用。

## 剩余差距：证据与限度

[最终逐阶段全表](stage-control-final-r2/RESULT.md)覆盖3KV×28阶段×两进程，输入均由同一原版前缀生成。中KV两进程中位数均值，单位µs：

|阶段|原版/guarded|自有独立（整层资源配额）|自有整层嵌入|
|---|---:|---:|---:|
|table|30.720|31.744|28.672|
|Self-Q|43.008|56.576|54.272|
|Self-attention|156.672|235.520|230.912|
|Cross-K|553.984|579.072|574.208|
|Cross-V|552.960|581.120|569.856|
|Cross-K RMS|69.632|116.736|116.480|
|Cross-attention|99.328|120.320|115.456|
|FFN1|166.656|202.752|198.912|
|GELU|15.360|30.720|29.696|
|FFN2|116.736|144.896|138.240|

独立stage外部marker包括kernel边界，嵌入时间包括本stage内部同步/device调用，差值不是纯grid等待；不能把各阶段中位数相加替代完整层时间。插桩使完整层增加0.40%–0.92%，正式表使用未插桩二进制。

OBSERVED：Cross-K RMS/GELU独立执行仍慢，说明并非全是跨阶段等待。[资源干预](resource-envelope-control/RESULT.md)保持同一stage函数：Cross-K RMS在允许更少shared、216CTA时约94.2µs，GELU在864CTA时23.6µs，比固定108CTA的约116/31µs更快，但仍没到原版。仅降shared、仍固定108CTA并非普遍有效。拆出的最小资源kernel是诊断对照，没有替换生产单层。

OBSERVED：最终完整层使用100096B动态shared、255寄存器/线程、静态localSize936B，API返回每SM可驻留1CTA。原版各阶段可独立选择tile/CTA和资源；单层统一资源配置限制轻量阶段，是有对照支持的部分原因。255/936这些静态数不等于动态spill字节，也不直接证明HBM瓶颈。

HYPOTHESIS：函数边界保存/恢复、attention组织和GEMM数据复用仍有代价。旧版本有NCU动态local证据，但不能将旧字节数冒充最终版。最新NCU首次窗口被外部进程打断，0kernel数据；新有限窗口等待，取得前相关细项UNKNOWN。

## 负结果与停止规则

较宽FFN1 K64把整个shared增至148000B，整层反而退化；小投影照搬原版M128×N128×两段K并行也退化；一warp/行RMS、两warp/行RMS虽有限精度通过，完整层均变慢。全部保留原JOB/结果/DECISION。不能从某个独立stage变快直接推断整层变快。

12个实现候选及08b正确性修复已结束，最终选择11，不追加架构搜索。NCU仅等待至原09:00截止，外部进程不处理。资源账本包含失败、准备、等待、清理，CPU构建与GPU阶段分开；见RESOURCE-AND-STOP-REPORT.json（当前若仍有等待则标明进行中）。

## 下一步

1. 完成最终版NCU后，将dynamic local、Tensor pipe和stall与上述stage表对应；无计数则保留UNKNOWN。
2. 依据当前剩余约11%–12.6%强基线差距决定下一轮：优先函数live state、self-attention在单层资源下的组织、Cross-K RMS和FFN写回。单层统一资源有真实代价，不能保证“合成一个kernel自然不慢”。
3. 这些原因明确后，再登记依赖就绪tile补位；不直接开始30层融合或跨阶段DAG，不以DAG掩盖原语成本。

四项科学权限仍false，production_admitted=false。未获得新的闭环成功率、有效提交成本ΣT_round/Σg或机器人E2E结论。
