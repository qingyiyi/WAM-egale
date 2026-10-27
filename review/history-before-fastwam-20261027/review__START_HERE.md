> 2026-10-25 HW已精简：从[项目入口](../.pipeline/INDEX.md)按主题读取，[维护结果](../.pipeline/maintenance/hw-compact-20261025/RESULT.md)记录边界与核验；历史原文保留。GitHub作业状态是发布快照。

> 2026-10-25持续论文协作：[Prompt](PAPER_COLLABORATION_PROMPT.md)、[当前方法/证据/缺口](PAPER-CURRENT-CONTEXT.md)。先记录本页所在commit；先理解再按用户指定逐段起草，不直接生成论文。

# 当前研究入口（2026-10-25）

先读[5090五环境投机速度与清理结果](CURRENT-RTX5090-SPEC-SPEED-20261025.md)，然后按该入口读取结果、审计和实际源码。请固定本文件所在commit，原主张不能被后续main覆盖。

当前：44完整三臂组的有限工程成本；TF/PF2未获全局质量验收，Robo前分叉隔离FAILED。所有四字段保持false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。

- [短HW与当前队列](../.pipeline/PROGRESS.compact.md)
- [最新SM120 FFN结果](CURRENT-SM120-FFN-NEXT-20261025.md)：当前仍S；FFN1 mbarrier流水W三KV精度通过但整层慢约0.8%，未采用；原A及DSM8/C4/C2负结果保留，不是机器人E2E。
- [旧完整导航](START_HERE-before-M41-20261025.md)，paused72、原RESULTS.json和旧固定commit仍保留历史身份，不能当实时覆盖。

原始相机/KV/全部velocity、模型权重、编译产物和凭据不公开；不能假装网页端已访问本机raw。仓库是进度中转，不是实时作业runner。
