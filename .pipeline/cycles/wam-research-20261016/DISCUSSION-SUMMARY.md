---
kind: discussion-summary
cycle: wam-research-20261016
status: active
updated: '2026-10-27T10:11:52.929047+08:00'
---

# 已确认的当前决定

用户睡前授权三张A100继续实验，筛查无问题后自动进行全量，并要求最后汇报target/TF/TF+PF成功率、trace、新阈值/接受率及5090加速，持续更新HW。[Maintain夜间决定](../../memory/global/requirements/decision-FastWAM夜间三卡推进并在筛查通过后自动全量三臂评估.md)。它覆盖先前“小规模之外等待确认”，不把科学权限自动设为true。

最新原文：

> 只要不测速都尽量每卡两woker并行eval

[长期偏好](../../memory/global/requirements/preference-非测速任务尽量每卡两个隔离worker并行评估.md)：覆盖所有非测速采集与评估，单路须实际资源原因；可以两环境客户端共享一模型，环境/RNG/可变状态/输出隔离，测速保持独占。

FastWAM Base固定平台与十步/shift5、原版checkpoint/stats身份不变。目标为候选去噪状态→target并行验证→合法scheduler路径提交→真实成本及闭环质量证据。不迁入LingBot旧full512 reset bank/CFG B2/B14/末端t0额外forward。VA2部署资产复现边界不写成架构/量化未公开，也不否定VA1历史研究。

本会话只负责投机；用户另会话负责算子，当前要求完整persistent迁移且以原Torch eager精度为gold，直接改现有optimized；算子候选不能直接算本轮投机加速，也不能相乘倍数。5090窗口按[共享协调记录](../../experiments/fastwam-base-small-closedloop-20261026/COORDINATION.md)交接。

代码在HW之外，与LingBot同级；三处权重长期保留。新tau只用FastWAM CAL重新测定；旧Long约80%接受效果只用于历史速度模拟，不强制新模型接受。PF2按历史两完整投机chunk后一次canonical，不能恢复已偏离的机器人状态。全量官方面板含训练/选择暴露，不能改名独立TEST。

四权限仍false；保留失败、PARTIAL、原预算/截止、所有历史负结果及WAM固定参考；不租新卡、不提权/驱动/服务/计数器、不杀他人进程，BAAI写入仅<A100_RUNTIME>。只维护同一篇论文证据与受影响段落，不自动写正文或公开推送。

[整理前完整讨论与全部原文](history/before-fastwam-overnight-20261027-010743/DISCUSSION-SUMMARY.md)保留夜间授权全文、之前资源与方法变更和历史边界。

用户10/27追加要求先迁回TMA＋1T4M3E＋mbarrier并保持精度；已按原Torch eager逐字节完成，保留新研究实现。投影/FFN用此分工，norm/attention不强制统一；阶段grid同步保留。对应[Maintain](../../memory/global/requirements/requirement-FastWAM完整persistent迁移先精度后阶段性能.md)。

用户10/27 08:46补充原文：“补充一下，加速比只计算都成功的任务，不成功的不要计算”。按[Maintain新要求](../../memory/global/requirements/requirement-加速比只统计共同成功的配对任务.md)保存为约束：三臂主表统一采用共同成功交集，按测速原始观测源episode身份筛选，报N和条件性；全体质量、失败和原始计时保留。只重算CPU汇总，不重复占GPU；旧含失败速度主表由新口径替代。

用户随后要求“prompt部分你也可以采样之前的缓存方式加速，去掉影响”。按[文本缓存共同优化](../../memory/global/requirements/decision-FastWAM文本缓存作为三臂共同优化.md)执行：三臂同用，只缓存文本、proprio每次重新追加；独立版本、条件/动作/接受路径检查和真实测速，不直接减旧时间。用户询问VAE设备后已沿实际加载链确认GPU BF16，Base图像编码走eager；本会话尚未改VAE或算子。

算子有限误差评估、FFN2对照授权及未采用结果见[本次归档](history/before-text-cache-report-20261027-0932/DISCUSSION-SUMMARY.md)；未设生产容差，严格版仍默认。

用户10/27要求研究不局限WAM，面向flow-based robotic action models；FastWAM之后，时间和资源允许时补StarVLA-PI、StarVLA-GR00T。未启动新实验，当前预算与优先级不变。见[原文与Maintain记录](discussions/20261027-跨模型研究范围.md)。

用户算子会话询问prompt缓存后能否直接读取cross KV并明确同意执行；[决定](../../memory/global/requirements/decision-FastWAM每个chunk预计算CrossKV并公平比较.md)。仅chunk内分别B1/B7；双方共同缓存，不改生产/阈值。实际结果见[缓存收口](../../experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)，eager逐位通过不代表compile或机器人新版质量通过。
