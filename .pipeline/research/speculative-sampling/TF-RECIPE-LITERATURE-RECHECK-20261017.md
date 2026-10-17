# TF训练与接受机制：本轮结论的文献边界

核验时间：2026-10-17，北京。仅原文核查与建议，不触发新训练、接受器、架构或120/TEST。项目科学基线仍为`b0dfac3a02f65de33a1217246b83003615166640`。

|原文与版本|直接支持的事实|对本项目的限制|
|---|---|---|
|[TS-DP arXiv:2512.15773v1 §3.1–3.2](https://arxiv.org/html/2512.15773v1)|单层Transformer draft；共享target encoder及DDPM/DDIM scheduler；输出蒸馏加scheduler-aware normalized loss；target冻结。其接受/纠偏推导使用概率密度比、共享协方差的Gaussian条件及reflection-maximal coupling。|本项目TF64的17维输入没有显式图像/prompt cross-attention，含previous teacher/draft方向的隐式条件；不是相同接口。当前确定性flow、非均匀sigma与L2门不能继承其分布保持表述。该入口仅核实预印本，未据标题声明A会录用。|
|[FlowCast ICLR2026正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)|无需训练额外draft，以当前速度外推、按速度MSE阈值验证；分析full/speculative flow轨迹偏差。|这里的trajectory是生成去噪状态，不能推出LIBERO物理环境步数或成功率保证；不是Transformer训练效果的证书。|

## 与本轮实测结合

本轮固定TF-on的真实新增FIT11416行、CAL6805行，来自旧TF_CONT3000完整30局的实际到达节点；不是新TF-on自身on-policy。TF-cont/TF-on公平500更新及MLP-on对照均完成；更低局部CAL误差不能替代完整闭环验收。最新进展只看[RUNBOOK](../../experiments/tf-effect-closure-20261016/RUNBOOK.md)，不要把本页静态核查当运行状态。

两条已核验新增风险失败说明当前配方没有满足质量要求；它们不能单独证明网络容量不足、条件不足、误差累积或某一数值差异是根因。单层Transformer在他文中可用，也不等于本项目当前权重可直接放行。

## 建议及止损

本轮结束后按登记停止继续TF质量调参，不以更长训练、更大规模测试或租更多GPU替代失败解释。可以冻结TF64接口开展系统机制研究；当前权重不能作为已合格的最终机器人策略。

若后续另立前瞻研究，优先明确“接受分数→scheduler最终输出→物理命令→接触/夹爪阶段”的实际偏差是否能预测新增失败，再决定是否有依据登记唯一接受挑战者或条件扩展。现有日志未保存的方向/中间状态须标UNKNOWN，不能事后伪造。不得按失败task硬编码回退；不能把本文作为挑战者已获准或已有效的证据。
