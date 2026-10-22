# 当前讨论（2026-10-22T14:42:31.869291+08:00）

用户补充要求持续记录优化以备论文使用，已保存[长期要求](../../memory/global/requirements/requirement-持续记录优化改动与论文证据.md)，并建立[优化证据索引](../../experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。同步追问Cross K/V的计算与tile来源；这是源码解释，不表示启动新优化。

用户在逐阶段讨论后同意将调制表按需融合到消费位置，要求重新计时并更新完整表。已通过HW Maintain保存[本轮决定](../../memory/global/requirements/decision-调制参数按需融入消费阶段并重测.md)，constraint，仅M39；替代此前这一项仅讨论的状态，仍保留阶段同步。

本轮完成：有限精度/状态通过，短/中/长整体快1.005%/0.979%/0.942%，采用新版。收益约23–24us/层，未完全对齐强guarded。普通路径少2016tile/一次grid同步/15.75MiB中间表；debug为审计保留表，不用于计时。

下一步仍按用户逐条讨论，从Norm1与输入调制开始，不自动开始下一融合或恢复ready。GPU已清理，累计预算和科学权限不变。[结果](../../experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)。

用户最新批准先做RMS+RoPE有限融合，顺序替代Norm1讨论。执行见[本轮计划](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/PLAN.md)，保持阶段同步；GEMM+RMS与新warp流水仍仅讨论。
