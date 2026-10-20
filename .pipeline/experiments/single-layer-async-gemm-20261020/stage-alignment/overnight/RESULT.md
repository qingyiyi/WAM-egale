# 单层算子：先对齐，再调度

更新：2026-10-21T02:29:04.119658+08:00。当前**有限精度逐位通过，完整层性能尚未对齐Torch**。活动源码恢复为RMS向量搬运后的顺序版；没有继续跨阶段ready调度。baseline、原tile和LIBERO投机代码未改。

## OBSERVED：完整单层

远端A100-SXM4-40GB，GPU0独占计算，B14、每实例Q16、D3072/FFN14336，KV短/中/长288/1152/2160。每个进程内native/before/simple交错，同一真实输入；计完整uncached/eager单层设备时间，排除准备，未用插桩时间替代正式样本。两个独立进程均完整保留。

|进程|KV|native ms|选定版 ms|选定/native|
|---|---|---:|---:|---:|
|1|short|2.630656|3.865600|1.4694×|
|1|middle|2.406400|3.562720|1.4805×|
|1|long|2.840224|3.811248|1.3419×|
|2|short|2.126336|3.326464|1.5644×|
|2|middle|2.398624|3.560672|1.4845×|
|2|long|2.839088|3.815056|1.3438×|

来源：[进程1](rms_vector/TIMING-1.json)、[进程2](rms_vector/TIMING-2.json)。短KV出现1095→1410MHz状态变化，不能挑最好轮或跨进程只取最快分子/分母。仍不是强cached/Graph基线、30层、chunk或机器人E2E。

[有限精度门](rms_vector/SIMPLE-GATE-RESULT.json)：B2 action/B14 action/B2 video×三种KV，共9签名；输出、每格23个共享中间量、18有效KV状态转移、输入/RNG/重复均逐位一致，最大绝对误差0。仅证明已测fixture；不等于任意输入或生产质量批准。恢复源码重新构建后的[阶段复核](stage_selected/simple-stage-process1-RESULT.json)也逐位通过。

## OBSERVED：最新版中KV阶段

A是原生阶段；B是自有阶段独立执行；C是自有阶段嵌入插桩整层。B仍保留CTA任务队列、cooperative同步和整层98848B shared预留，并非已为每个原语单独调优的kernel。A/B用Graph标记区间，C用内核globaltimer；不扣marker开销。B、C的缓存历史、编译资源和时钟可能不同，**C−B不是纯barrier耗时**。本表用于定位，不与上表直接相加。三种KV全部原值及同SM周期时钟在[CSV](STAGE-SELECTED.csv)与原JSON中。

|阶段|A µs|B µs|B/A|C µs|
|---|---:|---:|---:|---:|
|table|30.72|79.36|2.583×|103.94|
|norm1|11.26|16.90|1.500×|17.41|
|modulate1|6.14|15.36|2.500×|16.38|
|self_Q|35.84|41.98|1.171×|51.20|
|self_K|35.84|40.45|1.129×|62.98|
|self_V|35.84|40.96|1.143×|57.86|
|self_Q_RMS|6.14|13.31|2.167×|16.38|
|self_K_RMS|6.14|13.31|2.167×|16.38|
|self_Q_RoPE|6.14|15.36|2.500×|16.90|
|self_K_RoPE|6.14|15.36|2.500×|16.38|
|self_attention|376.32|293.89|0.781×|328.19|
|self_out|35.84|44.03|1.229×|61.44|
|residual1|6.14|15.36|2.500×|24.06|
|norm2|17.41|20.48|1.176×|25.60|
|cross_Q|35.33|40.96|1.159×|57.86|
|cross_K|553.98|585.22|1.056×|794.62|
|cross_V|553.98|585.73|1.057×|787.46|
|cross_Q_RMS|6.14|13.31|2.167×|17.41|
|cross_K_RMS|73.73|121.86|1.653×|99.33|
|cross_attention|90.11|99.84|1.108×|127.49|
|cross_out|34.82|41.47|1.191×|59.39|
|residual2|5.12|12.29|2.400×|19.46|
|norm3|11.78|16.38|1.391×|21.50|
|modulate3|6.66|15.87|2.385×|24.58|
|FFN1|162.30|196.10|1.208×|219.65|
|GELU|15.36|52.22|3.400×|78.34|
|FFN2|109.06|153.60|1.408×|167.94|
|residual3|6.14|15.36|2.500×|23.55|
|layout|4.10|9.22|2.250×|8.19|

self_attention的A包括原历史KV pack，因此不能把整阶段优势写成纯FA2核心优势。B14始终为14个独立Q16，不是Q112。

## 当前为什么还没有对齐

1. **阶段实现仍有差距。** 独立执行就能观测到GELU、table、FFN2及部分Norm较慢，不能全部归因于跨阶段等待。B仍包含队列、同步及统一shared预留，因此也不能把B/A全称为纯算术差距。源码的逐元素搬运、tile领取和GEMM epilogue与原Torch/cuBLAS实现不同；复用CUTLASS主循环不等于复现cuBLAS全部执行组织。
2. **统一kernel采用统一资源配置。** 选定完整层运行时是256线程、255寄存器/线程、98848B shared、1驻留CTA/SM、108CTA。小Norm和pointwise也在这个配置中执行；这与各自独立原生kernel的资源配置不同。静态/运行时容量已测，动态occupancy、spill、HBM/L2流量仍UNKNOWN，不能给它们编造耗时占比。
3. **单纯提高驻留量已经被反例否定。** compact B14把线程/流水/共享空间改为128线程、两级CUTLASS、66080B shared，达到2CTA/SM，但完整层反而慢约2%–10%。在中KV的独立阶段中，FFN2约286µs、self Q约78µs；其attention更快。它是组合资源配置的结果，尚不能把退化单独归于两级流水或某一个因素。见[compact阶段](stage_compact_r2/simple-stage-process1-RESULT.json)、[两进程计时](compact_b14_r5/TIMING-1.json)及[第二进程](compact_b14_r5/TIMING-2.json)。该方案已撤回。
4. **严格BF16边界限制可直接套用的实现。** 当前GEMM保留原分段BF16累计/偏置顺序，Norm保持原归约，所有中间量仍有全局workspace边界。不能通过改变舍入或只检allclose来制造精度对齐；但这也不是“不可能对齐”的证明。

这是“当前实现未对齐”的证据理由，不是宣称单kernel在理论上无法达到Torch。尚未测得纯barrier、动态spill或缓存流量的独立贡献；这些保持UNKNOWN。

中KV的同SM周期/globaltimer区间显示cross K、GELU及FFN2的B/C平均频率都约1.40GHz，因此这些格的C/B差距不能仅由频率差解释。独立GELU仅36寄存器、cross-K RMS仅60寄存器，但仍预留98848B shared；插桩整层255寄存器/线程、1432B local帧，而独立cross-K/GELU为0B local。这里的local帧是编译/运行时属性，不是动态spill流量。

## 保留与止损

保留：pointwise/layout搬运、Norm IO、双行RMS、FFN1 M128、cross连续KV加载、B14 M64 attention、RMS向量搬运。后者完整层改善约1.1%–1.7%。

撤回：FFN1 N256（慢3%–4%）、point8192（慢约1%）、强制GEMM内联（无稳定收益）、compact B14（慢2%–10%）。grid_constant原已存在，取消重复试验，0 GPU。所有失败、负结果、资源消耗保留。

下一步仍限阶段对齐：优先GELU/table的原生向量化方式，以及FFN2的tile/分段epilogue组织；先同输入独立阶段核验，再完整层核验。当前不启动跨阶段任务调度，也不以异步潜在收益掩盖原语差距。若后续无法继续缩小差距，提交阶段绝对成本和资源证据，而非放宽精度门。

完整源身份见[SELECTED-RESULT.json](SELECTED-RESULT.json)，实现入口为[whole_layer.cu](../../../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)；链接以仓库根目录算子研究路径为准。四项质量/正式性能权限全部false。
