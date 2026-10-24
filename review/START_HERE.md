# 当前研究入口（2026-10-25）

先读[5090五环境投机速度与清理结果](CURRENT-RTX5090-SPEC-SPEED-20261025.md)，然后按该入口读取结果、审计和实际源码。请固定本文件所在commit，原主张不能被后续main覆盖。

当前：44完整三臂组的有限工程成本；TF/PF2未获全局质量验收，Robo前分叉隔离FAILED。所有四字段保持false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。

- [短HW与当前队列](../.pipeline/PROGRESS.compact.md)
- [最新SM120 FFN结果](CURRENT-SM120-FFN-NEXT-20261025.md)：PAIR2048x2不采用，保留SelfQKV四MMA A；不是机器人E2E。
- [旧完整导航](START_HERE-before-M41-20261025.md)，paused72、原RESULTS.json和旧固定commit仍保留历史身份，不能当实时覆盖。

原始相机/KV/全部velocity、模型权重、编译产物和凭据不公开；不能假装网页端已访问本机raw。仓库是进度中转，不是实时作业runner。
