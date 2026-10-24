# FFN2 split-K 独立审查

范围：只读检查源码与已回传结果；未运行GPU、未改实验源码。初审文件mtime为2026-10-24T00:35:14.691878+08:00；初审时第二个独立FFN2进程及实际public入口结果尚未回传。下方保留初审发现，并于2026-10-24T00:41:07.370064+08:00追加补审；本文件不替代进程清理与资源总账审计。

## OBSERVED

1. **数值与计时入口分开。** `precision_unique.py`比较27项捕获值（含输出）、debug/normal输出、有效KV、输入、RNG；三个真实KV均为`FINITE_EXACT_PASS`。`measure.py`先校验debug中间量，再以`debug=False`的正常二进制完成四臂计时；计时前后四臂最终输出均逐位一致。正常计时未使用阶段插桩二进制。
2. **提交计数完整。** 当前`whole_layer.cu:584`的`action_ffn2_splitk`按每行组partition-major领取672个唯一任务；前驱片段必先有owner，提交依赖只指向更小partition。`ffn2_partition64`与`ffn2_commit64`数值函数未改。debug实测四个领取计数均168、完成计数均24、96个commit均7。它们是debug终态计数，不是正常计时每一次调用的逐任务轨迹；normal最终输出另行核验。
3. **完整层比值方向正确。** 复算`reports/SPLITK/SUMMARY.json`的两进程中位数均值，`1−Own/A`为短5.40365%、中5.17433%、长4.47411%。两个进程每档均改善：短5.3328%/5.4740%，中5.1202%/5.2285%，长4.5831%/4.3652%。`Torch/Own`分别1.14932/1.62454/2.04971，不能把这些整层总收益全部归因于本次split-K。
4. **强baseline仍有差距。** `Own/Guarded−1`短−1.5320%、中+0.2176%、长+2.5620%。因此可说短KV略快、长KV仍慢；中KV约持平，不宜宣称稳定击败Guarded。双方eager、无Graph、同缓存构建与同fixture，冷CrossKV另列。
5. **独立FFN2第一进程口径可用，但不是纯GEMM。** `measure_ffn2.py`重放真实ACT，比较FFN2+bias+门控残差。Torch提前构建gate，自有在epilogue按原公式重建；自有输出/状态workspace预分配，Torch走正常eager分配，差异已记录。SplitK约0.13185–0.13271ms，Serial约0.18944–0.19022ms，同口径Torch约0.13308–0.13481ms；`Torch_region/SplitK=1.00583–1.01751`，支持有限样本下约持平，不支持明显超越。纯linear背景为0.11382–0.11578ms，不能拿它与包含残差的自有直接宣布纯GEMM对齐。
6. **独立probe不能直接代表嵌入FFN2。** 两probe均170CTA、256线程、100096B shared、168寄存器、0 local；整层为255寄存器、784B静态local。资源不同被明确记录，独立probe不能单凭耗时相减精确分解整层容器开销。其两臂共享相同提取的device数学主体，计时前后输出/输入/RNG及完整提交计数通过。

## UNKNOWN / 限制

- 初审的第二独立FFN2进程与public两种stream待回传项，已由下方追加补审关闭；其余限制不变。
- 无NCU计数器证据，未测得Tensor Core活跃度、实际spill流量或semaphore等待占比；静态资源不能直接归因动态瓶颈。
- 阶段时间是自有嵌入globaltimer区间（含阶段同步）与Torch独立eager区间，属于诊断，不能作为可加的精确分解。独立FFN2的三种输入来自三KV fixture，GEMM形状相同，不等于三种GEMM规模。
- 有限fixture逐位通过不能推导所有输入等价、完整30层通过或机器人质量通过。

## RECOMMENDATION

现有证据满足计划中的“三真实fixture有限正确性＋两独立进程三档正常完整层改善”，实际public两种stream亦已通过补审；支持将split-K保留为**研究版本**，资源清理及最终文件回传另按收口审计确认。不要将独立FFN2约1%量级优势写成稳定超越Torch；不要称整个FFN已对齐，完整FFN1+GELU+FFN2+残差阶段仍慢于Torch。四科学权限和`production_admitted=false`继续保留。

证据：`results/precision-data/RESULT.json`、`results/SPLITK-measure{1,2}-data/RESULT.json`、`results/standalone1-data/RESULT.json`、`reports/SPLITK/SUMMARY.json`、`measure.py`、`measure_ffn2.py`、`make_standalone.py`、`standalone.cu`。

## 追加补审：2026-10-24T00:41:07.370064+08:00

- `results/standalone2-data/RESULT.json`：`COMPLETE_EXACT_STANDALONE_FFN2`。三真实输入的Serial/SplitK debug F2、debug输出、normal输出以及计时后输出全部逐位一致，KV/输入/RNG与提交计数通过。第二进程SplitK约0.13212–0.13231ms，Serial约0.18915–0.19021ms，Torch同口径约0.13300–0.13448ms；Torch/SplitK为1.00591/1.00961/1.01789，复现第一进程“约持平Torch”判断。两probe仍为168寄存器、0 local，不能替代整层资源口径。
- `results/public-default-data/RESULT.json`与`results/public-custom-data/RESULT.json`：均为`FINITE_EXACT_PASS`；实际`whole_api.extension`未替换，加载`wam_whole_splitk_cuda`。默认stream ID为0，自定义stream为166976272；两者各覆盖短/中/长KV。
- 六个public case的捕获中间量、normal/reference及normal/debug、有效KV、输入、RNG全部通过。每例96个commit为7、四个领取计数168、四个完成计数24。上述为本次直接读取原始结果并逐字段核对，不沿用仅有status字符串的判断。
- 结论：初审中两项等待回传已解除，未发现新的采用阻碍；有限正确性、非生产权限、纯GEMM与FFN整体边界等限制保持不变。
