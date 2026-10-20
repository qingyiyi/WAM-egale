# 当前讨论与授权

2026-10-21最新用户指示：**先对齐，再做调度；无法对齐给可追溯原因**。活动源码恢复已通过三项阶段优化的顺序版本，停止新增ready试验；已结束试验保留。此指示优先于旧夜间ready窗口，GPU预算和RoboTwin链不变。

用户最新授权两线工作至10/21上午09:30–10:00，投机继续迁移，算子先阶段性能再tile调度。[夜间计划](OVERNIGHT-PLAN-20261020.md)。已按Maintain保存该范围和与旧截止的关系。


用户最新明确：GPU累计预算10小时；先测简易版，与复杂版同速则只维护简易版，并删除复杂活动代码；补充注释、逐阶段对原版，再看能否接近原版耗时。已按Maintain更新[简易实现偏好](../../memory/global/requirements/preference-实验算子代码简洁并将检查移到外部.md)与[预算决定](../../memory/global/requirements/decision-算子累计预算十小时.md)。

有限精度通过，简易无稳定退化，活动复杂路径已归档。整体仍比原版慢约71%–93%；当前只完成测量/解释，尚未实现完全对齐。依据A/B/C结果先pointwise/layout，随后cross-K RMS、FFN1/cross-attention，最后ready跨阶段调度。原BF16/归约/合法KV条件不放宽，检查放到外部。

RoboTwin固定采集继续：截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。原截止03:37:02和24h帽不变，本guard不自动TF训练。四质量权限false，旧失败/retention/清理缺口保留。baseline/投机/原tile/保护HW不改；仅单层persistent；BAAI限~/cxz，不改驱动/时钟或杀他人。原[讨论](history/simple-first-start-20261020/DISCUSSION-SUMMARY.md)可追溯。

当前执行补充：顺序版本后续阶段改动已逐项测量，仍未达到原版耗时，继续以证据解释差距；RoboTwin原Target48及第一阶段训练完成，parent首局0动作接口计数失败保留，修复后仅接续未启动17局，补跑该局须等明确答复。
