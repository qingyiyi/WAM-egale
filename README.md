# WAM-egale：TF draft文献与源码交接

本分支为2026-10-17早最新暂停快照：`wam-tf-draft-handoff-20261017-paused72`。固定本文件所在的新commit阅读；原`b0dfac3a`历史快照与main保留。

**新30采集/固定三条件训练已完成；DEV仅72/96，TF-on出现三条T成功/TF失败风险反例，当前权重/配方未验收。GPU已暂停清理，四权限仍false，不自动恢复或启动TEST。**

- [从这里开始](review/START_HERE.md)
- [给网页版的完整调研Prompt](review/TF_DRAFT_RESEARCH_PROMPT.md)
- [最新结果JSON](review/RESULTS.json)
- [本轮结果与五报告](.pipeline/experiments/tf-effect-closure-20261016/RESULT.md)
- [单文件阅读包](review/REVIEW_PACK.md)
- [HW短入口](.pipeline/PROGRESS.compact.md)
- [逐文件来源与变化](review/SOURCE_MANIFEST.json)、[遗漏原件](review/OMISSIONS-20261017.md)

这是研究代码/文本证据快照，不是完整可运行环境或已通过质量的机器人策略。没有模型权重、原始轨迹/相机/KV、编译产物、profiler二进制或连接凭据。HW指Hypo-Workflow研究记录；旧计划、旧运行状态和公开源码本身不是执行授权。

TF64结构/API暂冻作研究载体，MLP只作公平ablation。请先核原文/官方开源的条件输入、teacher目标与训练分布，再规划一次有限诊断；不要因为局部误差更小就声称闭环成功率保持。

第三方许可保留，见NOTICE及原LICENSE。
