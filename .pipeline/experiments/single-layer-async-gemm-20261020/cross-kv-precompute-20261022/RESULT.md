# 预计算Cross K/V：双方更快，但自有相对强基线差距变大

更新2026-10-22T16:37:20.064075+08:00。已在optimized接入每层每条件组预计算K（包含原RMS）和V；warm完整层跳过两投影和K RMS，保留其余数值/阶段同步。原始K0保留用于有限中间量审计。原Torch/guarded通过外部适配器获得同等最终K/V缓存，baseline源码未改。

## OBSERVED：公平完整层对照

|KV|自有不预计算 ms|guarded不预计算 ms|自有差异|自有预计算 ms|guarded预计算 ms|自有差异|
|---|---:|---:|---:|---:|---:|---:|
|288|1.205248|1.227264|-1.79%|0.957440|0.887040|+7.94%|
|1152|1.372928|1.348352|+1.82%|1.123840|1.006080|+11.70%|
|2160|1.385728|1.358336|+2.02%|1.140224|1.020928|+11.69%|


差异为自有/guarded−1，负数更快。不预计算双方均只计算两组条件；baseline对原attention所需B14的展开仍计入。预计算双方都不再做K/V投影与K RMS，baseline展开也移到冷构建。自有自身延迟降约17.7%–20.6%，所以不应为缩小百分比差距而保留重复工作。

预计算后比强guarded慢约70/118/119μs；相对原Torch静态dense-history缓存路径仍更快，但论文不能用较弱基线替代最强guarded。原Torch静态缓存三KV约1.132/1.923/3.153ms，完整七臂见[CSV](WHOLE-LAYER-TIMING.csv)、[汇总](GPU-SUMMARY.json)。

两个独立进程、每臂600次预热、24轮交错，每进程三档KV合计13170次完整层调用。无外部GPU进程，SM1410前后核对；Graph边界不含Python分配、条件分组与冷构建，所有臂相同边界。不是30层、机器人task或E2E结果。

## OBSERVED：冷构建与缓存容量

每个独立进程首个short case的prepare墙钟42.804/42.668ms，随后middle为1.351/1.337ms、long为1.321/1.323ms。包含CPU条件分组、GPU准备与首次调用成本，不是纯K/V GEMM时间；首次额外开销根因未单独profile。baseline展开到B14额外约0.715–0.730ms（首case），随后0.272–0.306ms。冷构建未计入warm表，不将其隐藏或线性外推30层。

自有两组K0/K/V常驻18MiB，其中12MiB供warm使用、6MiB为审计保留；baseline原接口B14缓存126MiB。不是两边内存布局相同，而是同样预计算最终K/V、各自使用既有合理接口；两种容量均报告。

## OBSERVED：精度与逐阶段

9签名81次、27结构108次、三KV缓存baseline6次：输出与23共有中间量逐位一致，输入/预计算缓存/有效self KV/RNG及合法B2缓存转移通过。profile修复后的三KV×两模式也通过有限精度及状态检查。仅有限真实fixture，四科学权限false。

[完整阶段表](STAGE-TABLE.md)给25数学阶段加layout/exit的耗时、tile及公式，warm三项移至冷构建。普通warm5304/5376tile，31次grid同步，uncached仍6200/6272tile、36次。额外profile记录每CTA阶段及grid等待；正常版计时独立，扰动原样报告，不用平均CTA等待直接解释GPU墙钟差距。

## 阶段观察

中KV warm诊断：六小投影合计285.70μs、Self attention232.96μs、FFN1199.17μs、FFN2139.26μs。它们是自有层内部的成本集中点，不是已测出的相对guarded差额分配。

FFN1共有112个tile、108个CTA静态领取。CTA0–3需执行第二个tile，其余CTA在出口等待；本次平均CTA grid等待78.64μs，而CTA0仅3.07μs且阶段总199.17μs，与尾波不均相符。不能把78.64μs当成直接可消除的整层墙钟收益。FFN2的CTA0阶段139.26μs，其中grid等待15.36μs；大部分记录在屏障之外，但其中还包含领取、CTA/warp同步及访存。

建议先核对FFN1分块/尾波及六个小投影，保留同缓存强baseline；按需补同原语边界对照后才确定剩余差额根因。尚未实施下一轮调度。

## 解释与后续

OBSERVED：缓存前原guarded分组适配器每次须展开条件K/V；缓存后这部分也移出warm路径。双方节省的工作不完全相同，因此相对差距变大并不表示自有缓存实现使自身变慢。

HYPOTHESIS：剩余差距来自尚未对齐的阶段执行、资源限制及同步组织的组合，不能仅凭此实验认定全部是grid barrier。当前阶段表可以指出成本集中位置，但精确对原guarded各子阶段的归因仍需相同计时边界的原语对照。

RECOMMENDATION：保留预计算作为双方共同工程基线，再按本轮最大耗时阶段逐项对齐；未获新任务前不恢复ready调度或改变数学精度。预计算是成熟工程手段，不单独作为MegaTransformer创新。

静态研究API：`prepare_context(context, precompute=True)`后调用完整层；条件/权重/device/dtype改变须重新prepare，并重录使用旧指针的Graph。没有自动生产失效管理，生产/投机未集成。

[源码差异](CHANGE.patch)、[源清单](SOURCE-MANIFEST.json)、[资源/失败/清理](RESOURCE-AND-STOP-REPORT.md)、[上一轮分组复用](../cross-kv-reuse-20261022/RESULT.md)。正常DSO `0f62944795c64213dfa7106b9707a2f6ec5bd0aa41d9d008e94c2794289a0e67`；旧版本与全部失败保留。两机研究入口已采用，本机未新做GPU计时。

LingBot-VA2原文的6.5倍来自BF16 PyTorch起点，累积蒸馏、FP8 TensorRT、FlashInfer及运行时优化，并明确使用Cross K/V预计算；不是固定BF16单层算子的纯加速。[原文核验](../cross-kv-reuse-20261022/LINGBOT-VA2-ACCELERATION-BASELINE.md)。
