---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T21:45:12.259125+08:00'
plan: PLAN.md
current: M39-RTX5090-EAGER-STAGES
next: 与用户按融合阶段表讨论FFN优先诊断；双A100继续原RoboTwin队列。
---
# 当前进度

5090当前ffn-bias-last三KV精度保持；本轮已完成原Torch/guarded/自有三方eager两独立进程计时。短/中/长Torch/自有0.8896/1.2810/1.6427倍；自有比guarded慢约27%。当前融合FFN约0.50倍、占自有阶段46%–53%；三个独立小投影0.68–0.71倍，Self QKV约1.06倍。阶段为诊断边界，非纯kernel比；无插桩整层另列，长KV guarded首进程IQR5.09%保留。

阶段表按实际融合归并为15计算阶段＋布局/退出两项，CrossKV冷构建单列；FFN重叠不拆成可加时间。正常数学源码未改，所有实验仅单层真实重放；四科学权限及production_admitted=false。新计时产物已回本机，owned退出、显存0；保守累计25426.413/36000秒，余额2.937小时。

双A100 RoboTwin仍按原160小时队列接续；本轮21:38只读确认两卡各有模型/模拟进程，没有改队列或重新统计成功率。上次完整检查点20:58为148组/444局，仅历史动态快照。

下一步与用户讨论：先隔离融合FFN的生产/等待/分段提交成本，再决定单因素优化；三个小投影第二优先。不自动扩DAG或改变数值顺序。

[回传](../../experiments/rtx5090-block-port-20261023/eager-stage-comparison/REVIEW_RETURN.md)、[完整融合表](../../experiments/rtx5090-block-port-20261023/eager-stage-comparison/STAGE-TABLE.md)。

完整此前进度：[本轮前PROGRESS](../../experiments/rtx5090-block-port-20261023/eager-stage-comparison/HW-before-PROGRESS.md)。原FAILED、精度修复、预算、期限及历史授权未改。
