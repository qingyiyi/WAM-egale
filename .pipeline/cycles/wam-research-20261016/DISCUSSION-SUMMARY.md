# 当前讨论（2026-10-22T16:37:20.064075+08:00）

用户先批准Cross K/V分组复用并先精度后速度，后明确要求自有也读预计算并输出完整逐阶段耗时。两项均已完成有限单层核验，RMS未与GEMM tile融合。预计算数学包括每层两组K投影、K RMS、V；Q及候选私有状态仍独立。

不预计算同分组时差距约±2%，双方缓存最终K/V后自有慢7.9%–11.7%，但自有绝对耗时降低17.7%–20.6%。须保留同缓存强baseline，不能为缩小相对差距保留重复工作。逐阶段来自独立插桩，扰动及CTA等待边界在报告中。

首次诊断漏记两处同步，曾过早怀疑旧阶段表；已向用户澄清并修复插桩，原FAILED保留。计算仍36次，warm去掉五次变31次。LingBot-VA2对比BF16 PyTorch起点、最终叠加蒸馏/FP8 TensorRT/FlashInfer，预计算已有先例。

[当前结果](../../experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)。没有更改生产、四科学权限或恢复过期RoboTwin。

用户最新反馈要求阶段表同时提供tile数和每tile工作量。已补齐源码推导口径，不用逻辑tile总数比较不同阶段算量；后续汇报沿用此格式。

用户最新纠正：展示相对Torch加速比，不需要绝对耗时表；tile与工作量保留。完整层比值已由同轮CSV核算；最新逐阶段缺少同条件Torch计时，明确未测，不拼旧数据。原Torch静态缓存与最强guarded必须分列。

最新决定：后续只与原Torch实现比较，已Maintain保存，旧guarded证据保留。用户继续讨论GELU为何1568tile以及Tensor Core/CUDA Core是否可在FFN流水中同时工作；源码与依赖分析见../../experiments/single-layer-async-gemm-20261020/GELU-TILE-AUDIT-20261022.md。两个epilogue融合及跨GEMM流水目前均为建议，未实施，不将追问写成设计冻结。
