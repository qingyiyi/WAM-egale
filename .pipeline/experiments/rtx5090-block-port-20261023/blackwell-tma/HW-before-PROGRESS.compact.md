# WAM当前短状态（2026-10-23T21:45:12.259125+08:00）

5090当前ffn-bias-last三KV精度保持；本轮已完成原Torch/guarded/自有三方eager两独立进程计时。短/中/长Torch/自有0.8896/1.2810/1.6427倍；自有比guarded慢约27%。当前融合FFN约0.50倍、占自有阶段46%–53%；三个独立小投影0.68–0.71倍，Self QKV约1.06倍。阶段为诊断边界，非纯kernel比；无插桩整层另列，长KV guarded首进程IQR5.09%保留。

阶段表按实际融合归并为15计算阶段＋布局/退出两项，CrossKV冷构建单列；FFN重叠不拆成可加时间。正常数学源码未改，所有实验仅单层真实重放；四科学权限及production_admitted=false。新计时产物已回本机，owned退出、显存0；保守累计25426.413/36000秒，余额2.937小时。

双A100 RoboTwin仍按原160小时队列接续；本轮21:38只读确认两卡各有模型/模拟进程，没有改队列或重新统计成功率。上次完整检查点20:58为148组/444局，仅历史动态快照。

下一步与用户讨论：先隔离融合FFN的生产/等待/分段提交成本，再决定单因素优化；三个小投影第二优先。不自动扩DAG或改变数值顺序。

当前Cycle：[PLAN](cycles/wam-research-20261016/PLAN.md)、[PROGRESS](cycles/wam-research-20261016/PROGRESS.md)。

当前证据：[融合阶段表](experiments/rtx5090-block-port-20261023/eager-stage-comparison/STAGE-TABLE.md)、[回传](experiments/rtx5090-block-port-20261023/eager-stage-comparison/REVIEW_RETURN.md)、[固定入口](experiments/rtx5090-block-port-20261023/eager-stage-comparison/SELECTED.json)、[资源](experiments/rtx5090-block-port-20261023/eager-stage-comparison/RESOURCE-AND-STOP-REPORT.md)。

原精度证据：[三KV26中间量＋输出及156次确认](experiments/rtx5090-block-port-20261023/precision/REVIEW_RETURN.md)。原迁移数值FAILED不改，新数值版不继承旧加速比。当前仅B14/Q16、block0、只读KV、静态fixture；不外推在线任意shape/30层/机器人质量。

机器人：[双A100交接](experiments/robotwin-pilot-20261020/eval100-20261022/dual-continuous-20261023/RESULT.md)。600组1800局扩大DEV，累计160GPU小时，登记10天截止11/01 22:03，guard更早10/29 14:03不改；无自动重试。微波炉100组齐后再诊断。LIBERO四套6000局历史齐，但训练/DEV暴露与严格配对等缺口保留，不能签独立TEST。

保护state/cycle/rules、原tile、baseline、投机生产代码；固定HW v15不全量sync。不做SHA/smoke、不提权/改驱动、不自动消费余额。BAAI所有项目工作限~/cxz。

日期：10/23冻结身份与claim；10/28主表；11/01 18:00中文证据包，证据不足收缩claim。当前预算与所有原失败/期限不覆盖。

[更早短状态](experiments/rtx5090-block-port-20261023/eager-stage-comparison/HW-before-PROGRESS.compact.md)。
