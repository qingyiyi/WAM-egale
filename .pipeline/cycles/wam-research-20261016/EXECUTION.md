# 当前执行检查点

## 2026-10-21T09:13:44.589629+08:00 M39 / M40

- M39：真实输入剥离并核实Torch实际kernel。vec4入选；并行K的FFN2独立约慢4%–6%，整层不受益而撤回。全部数据/源与PID清理收据已归档。
- M40：获准新窗口的valid08自然失败，parent18齐并审计；固定CHILD/CONT训练和每权重10771 query接口CAL完成。保留原PARTIAL、语法失败0更新和全部自然失败。
- 原budget和deadline不重置。当前两GPU无计算进程；未开新episode，不截短任务。
- 下一步：整层FFN2条件归因；新窗口接CURRENT18。科学权限仍false。
- [报告](../../experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md) / [历史](history/pointwise-isolation-20261021/EXECUTION.md)。

## 2026-10-21T09:48:51.968651+08:00 M39 / CPU耗时汇总

用户要求先看整体/分阶段绝对时间和源码。已生成[版本分明的耗时矩阵与归因边界](../../experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/TIME-BREAKDOWN-20261021.md)。当前整层与旧阶段数据分列；不能量化纯barrier占比。只读分析，无GPU/编译/重试，未扩展旧窗口。反馈合并入现有Maintain记忆。
