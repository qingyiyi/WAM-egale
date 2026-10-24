---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-24T01:40:36.304385+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前进度

本轮三个小投影有序split-K已完成并拒绝：两进程完整层相对A退化42.700%/40.873%/38.087%。候选26中间量＋输出/输入/KV/RNG及计数均有限通过，静态资源与A相同；退化集中于三个投影，尚未定量区分领取/等待/发布开销。候选及负结果保留。

当前实际保留SPLITK-M64-FFN2，已恢复五源文件及原public入口；默认/非默认流再次通过，六份源码/DSO回本机byte一致。当前正常完整层原Torch/自有1.1505/1.6270/2.0413×；相对guarded短快1.889%、中快0.171%、长慢2.987%。全部eager，无Graph，同缓存。完整17阶段已本轮重测：三个投影约0.67–0.70×，融合FFN0.864/0.858/0.865×；当前和拒绝候选各自完整表保存、在对话打印，非历史混填。

本轮11469/15000完整block，两个构建/六GPU进程成功、无timeout，性能拒绝1个。owned16身份退出、5090显存0；保守计15分钟，累计32386.413/36000秒，余1.004小时，旧deadline不重置。四科学权限和production_admitted=false。

小投影每K段仅5次K64迭代（末段3次），比FFN2每段32次短；细粒度调度难摊薄开销是待定量机制，不能说已确定atomic或fence占比。下一步建议先分量诊断，再考虑每任务多段的一个粗粒度对照；尚未启动。

双A10001:28只读检查模型/模拟上下文和CPU巡检仍在原RoboTwin队列，本轮不改队列、不重算成功率。

[结果](../../experiments/rtx5090-block-port-20261023/projection-splitk/REVIEW_RETURN.md)、[当前完整表](../../experiments/rtx5090-block-port-20261023/projection-splitk/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/projection-splitk/SELECTED.json)。
