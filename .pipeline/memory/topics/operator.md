# 算子与源码讲解

先读[现行约束](../CURRENT-CONSTRAINTS.md)。以下按任务选用，不自动逐项全读；当前数字读Cycle。

- [整层单kernel与tile职责](../global/requirements/requirement-单层Transformer统一persistent内核调度全部tile.md)
- [逐段带读、完整阶段表与验证边界](../global/requirements/requirement-算子交付明确源码入口与验证范围.md)
- [代码边界与原tile保护](../global/requirements/decision-算子代码只维护baseline与optimized且不改投机侧.md)
- [先数值后性能](../global/requirements/decision-5090迁移先对齐精度再调优性能.md)
- [原Torch对照](../global/requirements/decision-算子后续仅对比原Torch实现.md)
- [保留阶段同步的当前边界](../global/requirements/decision-回退阶段全局同步并重新规划tile调度.md)
- [七段FFN有界接续](../global/requirements/decision-SM120七段FFN先诊断提交成本再有限合并owner.md)
- [双CTA实验范围；已完成负结果按Cycle读取](../global/requirements/decision-优先验证双CTA混合DSM并分离慢因.md)

当前采用S（FFN两阶段同步版），原A及C2负结果保留，实际数据见Cycle。10/25已授权并完成两阶段修改/测量；10/25随后批准仅FFN1的1T/4M/3E有界试验，采用由精度/完整层结果决定；其他阶段不统一warp比例，不新增DSM。历史30层融合、旧GPU时段和旧候选不自动恢复。
