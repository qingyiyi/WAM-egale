# 完整原生 Transformer block：有限复现已完成

终态：FINITE_NATIVE_BLOCK_PARITY_AND_TWO_PROCESS_DYNAMIC_TIMING_COMPLETE。两份入口均执行完整原生 block 数学；已在实际模型中安装全部30层并完成六组 action query 的严格比较。当前结果支持有限真实输入下可替换、动态 eager block 性能不回退；尚未取得最强 shared-prefix/缓存 Graph、完整新 policy/机器人闭环或正式 E2E 证据。

## 实现与问题修复

baseline 和 optimized 共用完整 C++/CUDA 顺序链：LayerNorm/RMSNorm、sigma 调制、Q/K/V 与输出投影、FP64 运算的 RoPE、self/cross attention、残差、FFN、缓存读写/驱逐与候选打包。FFN/各投影直接调用成熟 cuBLASLt，attention 使用本地固定上游 Flash CUDA。Torch 仍提供 Tensor、allocator、stream、handle；不转调 ATen linear、LayerNorm、SDPA 数学。这里是本地可编辑实现，包含有来源及许可证的上游代码，不声称从零自研 GEMM/Flash 算法，也不是一个大融合 kernel。

首轮 scalar 版完整动态 block 的 B14 中/长约慢3%；独立 profiler 定位到标量 KV pack 与串行 slot 扫描，middle/long B14 pack 每块约1566/3023微秒，layout约315微秒。最终 v7 改为16字节原样复制和 CUB 按 physical order 稳定 compaction；保留原 dtype、可见性和全部数值门。GEMM/attention 不是这次 scalar 回退的主要差异。该修复是工程基础，不能包装为新的 MegaTransformer 机制。

optimized 仅在 Q16 使用已验证的 FFN1＋GELU 融合，保留中间 BF16 舍入；Q128与 baseline 同链。FFN2 使用匹配原 Torch descriptor/heuristic 的 cuBLASLt；旧 CUTLASS FFN2 逐位失败原样保留。当前原生动态 KV 长度仍有16字节 host copy＋stream sync，CUDA Graph 暂不支持。

## 精度与状态覆盖

| 门 | 实际覆盖 | 终态与证据 |
|---|---|---|
| block关键量 | KV288/1152/2160 × B2/B14 action，逐点23关键中间量/输出、6个原语 | 六点逐位 PASS；native-vector-intermediates-final/results/RESULT.json |
| KV/驱逐 | action Q16、video Q128，各短中长 read0 与连续 write1→write2→read0 | 24门 PASS，含真实2160容量驱逐；native-vector-state-final/results/RESULT.json |
| optimized写状态 | Q16短中长连续写读 | 12门 PASS；native-vector-optimized-states-final/results/RESULT.json |
| actual30 action模型 | 六query × original/baseline/optimized；模型全部30层替换 | 最终18完整模型调用、42数值/状态检查，逐位 PASS；native-original-gradflags-final-attempt2/results/RESULT.json |
| canonical参数接口 | 原模型839个requires_grad=True标志保留，仅推理模式下用只读detach别名 | 输出、所有有效KV、metadata、输入、RNG、参数地址、安装还原通过；同上 |
| 计时中的一致性 | 两进程，各9点四引擎、每轮输出/有效状态比较 | 72组计时前检查、全部计时末输出/状态逐位通过；两个isolated-process结果 |

全部有效 KV 在现场逐位比较；没有把30层完整 KV 复制成新的离线数据集。有限输入 PASS 不是所有输入/硬件/版本的数学证明。

Q128输入来自三个既存完整 canonical target policy 的恢复重放：三次各72 forward，另18次prefix恢复，共234次模型调用，无新机器人 episode。其五个既有保存关键张量逐位一致。B14实际30层是旧真实 verifier 输入；B2实际30层由同一root的CFG行0/7派生，不能冒充保存的 canonical sequential B2 trace。视频只验完整 block0及缓存状态，没有宣称视频30层或新native完整policy已通过。

## 完整动态 block 性能

两次独立进程均完整跑9点，先无可见外来 GPU PID，过程中持续核查；每点 original_A/baseline/original_B/optimized，7轮轮换顺序，每样本5次、每引擎2次warmup。安装/编译/加载在计时外；真实 KV 管理、pack、同步在计时内。无 profiler/NCU、无时钟修改。GPU 为本机同一 A100-SXM4，CPU固定4/5两线程。

原始504条样本见 PERFORMANCE-ALL-RAW.csv；两进程各自的9组中位与wall见 PERFORMANCE-PER-PROCESS.csv。便读表见 [PERFORMANCE-TABLE.md](PERFORMANCE-TABLE.md)：毫秒列是两个进程中位的算术平均，比例列保留各进程范围，不将它当置信区间。

六action点 baseline/Torch 的两进程中位比范围0.4989–0.8028，optimized/Torch为0.4914–0.7880；三video点各为0.6345–0.7404与0.6190–0.7382。18个process×case里，两实现相对 original_A 和 original_B 的中位都不慢。此前“B14慢2.5%以上”不出现在这个最终完整native动态矩阵。

但 A/A 最大偏差8.63%，short-B14跨进程本身也有较大变化，故不将 optimized 相对 baseline 的小差值写成稳定新增收益，不宣称精确到1%的加速比例。较大的 native/Torch 差值含 Python 调度/动态元数据管理的工程收益，不能解释成纯attention/GEMM加速。

这一表比较原始完整动态 Torch block；M26最强shared-prefix/缓存projection/Graph是另一执行边界。旧表不混入，当前不能据此说胜过最强B14或得到MegaTransformer增量。30层本轮只验证正确性，没有测完整模型速度；更没有新chunk、episode或E2E速度。

## 资源、旧失败和交付

23个有身份收据的GPU子作业累计1088.541/3600生命周期秒（约18.14分钟），含等待/失败/准备/清理，CUDA active未测；288/360完整模型forward，5114/12000独立block意图、144/20000独立原语。完整模型内部另派生8640次block执行，单独披露，不与独立调用计数混称。实验原件约159MiB，低于1GiB。所有owned GPU worker均真实退出，不杀他人、不改服务或时钟。完整原始账本 [RESOURCE-AND-STOP.json](RESOURCE-AND-STOP.json)。

保留2次build失败、视频旧文件缺字段、首次模型混合dtype契约拒绝，以及外来GPU/等待到期等全部旧attempt；后续在独立namespace补齐。失败不改为成功，旧M21/M25/M26/M27门不改。

源码讲解与完整路径见 [CODE-GUIDE.md](CODE-GUIDE.md)，最终源/二进制身份见 SOURCE-DELIVERED.json。生产投机源码、配置、运行依赖、原tile与保护文件未修改；四项质量/TEST/非劣/正式性能权限仍为false。

建议下一步固定此完整baseline，先与现有最佳B14 shared-prefix/缓存projection/Graph对齐执行边界，再选择一个由profile支持的候选感知子图。跨候选attention共享必须保持各自Q16与private KV边界；不能只靠普通batching、vector copy或常规GELU融合写创新。
