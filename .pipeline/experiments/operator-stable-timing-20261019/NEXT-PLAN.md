# 按网页方案评估的算子下一步

状态：PROPOSED，CPU审查完成；本轮未启动新GPU/训练/机器人作业。科研参考仍ca5f37e，10月19日新结果作为独立增量，不回写固定快照或旧FAILED。用户要求判断计划，不视为已批准具体原型或新预算。

## 当前判断

原方案主线是完整block内部的依赖感知tile调度。现已具备可独立调用的混合顺序block和有限GPU数值/状态证据；当前optimized只将self/cross SDPA换成独立CUDA，投影、norm、FFN仍为Torch库调用。它们不是可直接嵌入persistent kernel的device函数。

两进程新验证108项逐位通过；block baseline/原版中位比0.9710–1.0272，完整30层0.9539–1.0058；optimized完整30层0.9535–1.0218。旧±2.5%失败/新筛查非全点、空卡NVML24–25%未知保留，性能未获正式无争用证书。已有证据足以继续CPU设计和有限热点定位；进入scheduler收益实验还缺强基线可比较性、真实profile和可对齐device GEMM。

本轮不再重写整套block或重复无新增问题的大矩阵；只补与下一决策直接相关的证据。

## 唯一默认路径与分阶段交付

|阶段|工作和交付|推进门|计划工作量，不含GPU排队|
|---|---|---|---|
|1，10/19|冻结baseline，补当前两份代码覆盖表、真实DAG/显式舍入与强B14组合兼容表；只加载一个block所需checkpoint参数，复用最多六真实fixture|R0/R1/B0身份清楚，cache/Graph适用性可核验，原dtype/布局/有效KV契约未变|CPU≤2，约1–2小时；GPU0|
|2，10/19–20|先A/A与时间线，再对热点采计数；B2/B14×K288/1152/2160，分离QKV、self/cross、context投影、FFN、norm/调制、pack/copy、host launch；prepared与完整接口成本分列|有稳定可解释的占比与实际GEMM M/N/K；强路径可比较，能给候选子图收益上界|建议新GPU生命周期≤30分钟，分析约1–2小时|
|3，10/20|只选一个热点子图。FFN1→原BF16 GELU→FFN2行组链是默认备选；先验证可在device内调用且精度/成本合格的GEMM/row-op|D0输出和中间舍入过原契约；设备资源/同步约定明确；重写GEMM损失未吞掉可省成本|实现约0.5–1个工作日，GPU试验另登记|
|4，10/21–22|同一D0做P_seq与P_sched，最多两个调度配置；先完整block，再仅对胜出版本测30层/允许范围内状态回归|P_sched对P_seq有可复现增量，并超越R_strong；否则收缩/停止调度主张|约0.5–1个工作日；样机阶段合计建议≤7 GPU小时、最多两个初始开发工作日|

以上是建议上限与日历，既不新增租卡也不重置已终止账本。实际job启动前沿现有授权登记范围、输入、绝对截止和owned清理；GPU等待会延后墙钟。10/22仍无关键证据就给NO-GO/明确缺口，10/23只冻结已有接口/claim，不承诺样机一定成功。

## 公平对照和收益归因

R_strong是预先固定、精度已在对应范围通过的最强可兼容B14 shared-prefix/O3/Graph/cache组合，不是任意名称叠加。兼容性未知就标UNKNOWN；不把当前无context缓存的原block当作唯一强基线。

profile同时记录uncached与既有合法cached边界；正式新调度增量对照中双方固定同等cache、Graph、workspace与准备开销。CPU提交问题先与合理Graph路径比较：PyTorch官方说明Graph主要降低CPU分派开销，不能将这项普通工程收益算成新tile调度。

必要三臂：R_strong；P_seq（相同device原语严格阶段顺序）；P_sched（相同device原语/数学配置，只变ready任务派发）。P_sched胜P_seq但仍慢R_strong，不算系统胜出。

核心六fixture已有单状态/七状态形状。device/GPU级实际Linear扩行仍待profile，不将CPU逻辑M32→224当成已实测GEMM。机制成立后再加一个真实合法三状态诊断点，缓存固定，检查新路径是否在候选扩行时获更多收益；两形状同幅收益只支持通用优化。每个候选仍独立Q16，不合并私有KV或跨候选注意。

## 首个子图为何优先考虑FFN

FFN链可在block内隔离，不需要同时改变attention可见性/KV协议，适合检验候选行组并行和生产者/消费者推进。是否为时间热点仍UNKNOWN，以强cache路径profile决定；不按参数量或FLOPs预判。若收益上界很小，停止该备选。

FFN2某M行组需要14336维activation归约输入就绪，不能FFN1一个N tile刚结束就发布完整ready。保留GEMM输出到BF16、原GELU tanh再到BF16和后续FP32 gate/residual边界；少写一次global中间tensor不代表可以省略原舍入。首版仅静态DAG、M/N输出tile与轻量ready计数，不引入跨CTA split-K；就绪发布/消费和workspace复用须有明确同步，不能使所有resident CTA等待尚未被调度的生产者。

GEMM优先核对现有SM80 CUTLASS组件；cuBLASDx0.5官方支持内核内GEMM及SM80 BF16/FP32 MMA，可作另一个有条件备选。但它依赖commonDx及CUTLASS4.2，本地兼容性/数值/性能尚未验证，不自动安装或覆盖现有vendor。D0仍可能因累加顺序不同无法满足逐位契约；最多两次有依据纠错，失败就停该路线，不通过改tau/容差/精度放行。

## 终点与停止线

本阶段目标是一个真实完整block中可归因、超过强基线的调度收益，不先30层大融合。已有工程筛查建议为30层新增改善约5%、完整chunk约3%，只作为前瞻投入门而非改旧性能或质量标准；样机微小改善/噪声内结果不扩大主张。

后续完整verifier、合法policy更新、全部有效KV/RNG和原质量门逐级验证；正式TF性能受原放行限制。最终回到sum(T_round)/sum(g)和完整chunk/任务成本，单GEMM加速、occupancy或target调用下降均不能替代。所有四项false、投机源码、原tile与保护state/cycle/rules不改。新代码只在optimized，共用诊断工具可补，baseline固定，不新增第三份活动实现。

证据：[实际新复测](RESULT.md)、[当前代码](../../../算子研究/README.md)、[原契约](../wam-block-baseline-20261019/BLOCK-CONTRACT.md)、[原readiness](../wam-block-baseline-20261019/SCHEDULER-READINESS.md)、[源清单](WEB-PLAN-REVIEW-SOURCE.json)。

官方方法参考：[PyTorch2.9 CUDA/Graph](https://docs.pytorch.org/docs/2.9/notes/cuda.html)、[CUTLASS4.2.1 grouped scheduler](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/grouped_scheduler.html)、[cuBLASDx0.5要求](https://docs.nvidia.com/cuda/cublasdx/0.5.0/requirements_func.html)。官方库存在不等于本地已实现或与Torch逐位一致。
