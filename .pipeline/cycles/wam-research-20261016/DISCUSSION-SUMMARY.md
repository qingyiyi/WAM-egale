# 当前讨论与授权

用户最新明确：GPU累计预算10小时；先测简易版，与复杂版同速则只维护简易版，并删除复杂活动代码；补充注释、逐阶段对原版，再看能否接近原版耗时。已按Maintain更新[简易实现偏好](../../memory/global/requirements/preference-实验算子代码简洁并将检查移到外部.md)与[预算决定](../../memory/global/requirements/decision-算子累计预算十小时.md)。

有限精度通过，简易无稳定退化，活动复杂路径已归档。整体仍比原版慢约71%–93%；当前只完成测量/解释，尚未实现完全对齐。依据A/B/C结果先pointwise/layout，随后cross-K RMS、FFN1/cross-attention，最后ready跨阶段调度。原BF16/归约/合法KV条件不放宽，检查放到外部。

RoboTwin固定采集继续：截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。原截止03:37:02和24h帽不变，本guard不自动TF训练。四质量权限false，旧失败/retention/清理缺口保留。baseline/投机/原tile/保护HW不改；仅单层persistent；BAAI限~/cxz，不改驱动/时钟或杀他人。原[讨论](history/simple-first-start-20261020/DISCUSSION-SUMMARY.md)可追溯。
