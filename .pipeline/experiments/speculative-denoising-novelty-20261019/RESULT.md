# 投机去噪重合核验与本项目可保留的研究问题

性质：CPU只读研究，未新增GPU作业/训练/评估。核验日期2026-10-19；固定论文版本和定位见 PRIMARY-SOURCES.json。本报告是相关工作与研究判断，不是新论文初稿。

## OBSERVED：大方向和多个组成部分已有直接先例

| 原文与身份 | 已有内容及定位 | 与本项目重合及边界 |
|---|---|---|
| [Accelerated Diffusion Models via Speculative Sampling，ICML2025](https://proceedings.mlr.press/v267/de-bortoli25a.html) | 连续空间Markov采样的draft/target投机框架，含Frozen Target Draft | 连续扩散投机不是空白；其随机转移精确性不能自动转给本项目确定性Flow Euler＋局部L2接受 |
| [TS-DP，arXiv v1，2025-12-13](https://arxiv.org/html/2512.15773v1) | §3.2单层Transformer draft、冻结teacher、预测MSE＋scheduler-aware归一化损失、一次batch验证多个去噪状态；§3.3 PPO调度 | “Transformer draft＋蒸馏＋机器人去噪＋并行verify”已被直接做过。其共享encoder、DDPM/DDIM与随机接受不同，不能当成当前TF64质量证明；4.17×表值对应NFE口径，不能当任务E2E |
| [FlowCast，ICLR2026正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html) | §4式5和图1：常速度预测，velocity MSE门，单次并行验证、合法前缀与拒绝后重启；§4.1有假设的ODE误差分析 | 连续Flow的velocity误差验收、零训练draft和batch去噪已有。其图像/视频轨迹误差界不是机器人环境步数或成功率保证 |
| [Realtime-VLA FLASH，arXiv v1，2026-05-13](https://arxiv.org/html/2605.13778v1) | §3.3单Gemma block约110M参数直接回归动作chunk；§3.4在若干插值时刻并行重建终点验收动作前缀；Table3周期PF与FB分开消融 | PF和小Transformer动作draft均非新；它跳过VLM prefill并复用旧上下文，粒度与本项目chunk内去噪树不同。PF2单因素80.6%、FB+PF2为84.6%，不可混用；本项目SST仅action全target，非其full-path刷新 |
| [ParaDiGMS，NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)；[ParaTAA，ICML2024](https://proceedings.mlr.press/v235/tang24f.html) | Picard/固定点并行去噪；前者包含机器人DiffusionPolicy | “串行去噪变并行、额外计算换延迟”也非新；不是当前learned draft树的同一算法 |
| [Sequoia，NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html) | 树结构DP与批量验证；较新的[arXiv v3](https://arxiv.org/abs/2402.12374v3)明确hardware-aware树优化 | 不能宣称首次用投机改善GPU利用率、树批量验证或按硬件成本选树；区分会议版与后续版本 |
| [Cascade，FlashInfer官方资料，2024-02-02](https://flashinfer.ai/2024/02/02/cascade-inference.html) | 公共prefix/私有suffix分解、共享prefix batch decode和attention统计合并 | 普通shared-prefix缓存、拆分softmax合并不是新机制；存一份KV也不等于只读一次HBM |

TS-DP与FLASH本轮只确认了预印本原文，正式录用会议UNKNOWN，不把它们写成已核验A会。上表正式会议身份来自对应主会论文集。没有借标题推断算法，也没有把NFE或target调用比例当实测任务速度。

## OBSERVED：新增需要纳入的近邻

[Speculative Forcing，2026-10-15 v1](https://arxiv.org/html/2610.22347v1)把中间clean估计当draft，按token收敛提前停止去噪，保留已接受token的KV；包括Lingbot-World2.0。它是视频/可控世界生成，不是LingBot-VA机器人动作成功率，但“世界模型中的投机去噪”不能再笼统声称未被研究。

[WAMJET，2026-10-07 v3](https://arxiv.org/html/2610.03797v3)已包含LingBot-VA，§IV-D明确涉及prompt/KV复用、Graph、projection、norm/modulation/activation/residual融合；主张是agent优化框架。仅复现完整block、普通缓存、减少Python开销或常规融合不足以单独形成我们的系统创新。其跨硬件速度不能搬来与本机数据直接比较；其容差/精确性定义不覆盖本项目逐位契约。

[RIFT，2026-09-25 v3](https://arxiv.org/abs/2608.11521v3)研究用learned anticipation tokens一次构建future KV，避免迭代video rollout；[1Step-WAM，2026-10-11 v1](https://arxiv.org/abs/2610.15625v1)学习初始velocity校准并做one-step action，官方[仓库](https://github.com/changdong1687/1Step-WAM)本轮可见仅README/图。它们是不同路线的WAM近邻，不能因机制不同就不比较质量/速度，也不能据README保证可在当前checkpoint直接复现。

[WA-SpecDec，2026-08-09 v1](https://arxiv.org/abs/2608.08725v1)针对自回归VLA token投机，加入world-aware prefill；不是本项目连续velocity去噪树。相似名称不等于相同实现。

## 本地OBSERVED与UNKNOWN

当前TF结构见 投机采样/experiments/draft_architecture_v1/models.py:39；冻结loss实际调用见 tf_effect_train_v1/train.py:278。prefix_then_cost选择见 prefix_acceptance_v1/selection.py:15。当前验证入口、scheduler、条件/私有KV边界保持冻结；不能为了与文献对齐而在本轮自动换loss或架构。

M28完整native顺序block、只读参数别名、16字节KV复制/CUB稳定compaction、FFN1＋GELU均已取得有限正确性/动态block计时证据，但尚无超越最强B14 shared-prefix/O3/缓存Graph的证据，也没有新native完整chunk/episode速度。四项质量权限仍false。

本轮没有克隆/执行外部实现，所有外部源码commit/hash均UNKNOWN；论文与官方README核验不冒充代码复现。检索不是全领域穷尽证明，不能从“这些文章未测某接口”推出“我们全球首次”。

## HYPOTHESIS：可能属于本项目的增量

可以研究：冻结联合世界动作模型中，连续候选带来共享历史/条件与各自sigma、私有KV的结构，如何通过候选感知Transformer子图/tile调度，在已有最优B14上进一步降低有效推进成本。这里的贡献应是具体可复用执行机制和可验证因果证据，不是“把batch从2改成14”。

需要保留每个attention实例Q16；跨候选组query tile时只复用实际相同的合法prefix，不能把7候选变成互相可见的Q112序列。线性层M32→224通常已由库展平实现，不能重复申报为新算法。

## RECOMMENDATION与主张门

| 主张 | 当前是否可写为成立 | 还缺什么 |
|---|---|---|
| 首次用小Transformer/蒸馏/并行verify加速机器人扩散 | 否，有直接先例 | 应引用并收缩主张 |
| PF2或velocity阈值本身是新方法/成功保证 | 否 | 原始接受/调度已有；闭环质量需独立证据 |
| 有限真实输入下完整native block可复现原Torch | 可以，限M28声明范围 | 任意输入/生产完整闭环不在证据内 |
| candidate-aware执行超越现有强B14 | UNKNOWN，不能当结果 | 1/3/7形状×现有/新引擎、shared-prefix与cache因素分离 |
| 投机与系统调度有额外交互 | UNKNOWN | 同口径因果消融、profile和稳定延迟；统一图例定义，勿混淆不同文件中的A/S含义 |
| 闭环质量保留并获得E2E收益 | 尚不允许 | 独立TEST、质量门及同硬件无争用完整计时；同时报SR/全失败罚分/长度 |

下一步默认：固定刚完成的完整native baseline → 与现有最佳B14建立可比强基线 → 仅选一个有profile依据的候选感知机制 → 记录sum(T_round)/sum(g)、完整chunk与闭环结果。不新增architecture sweep、不直接大融合、不把论文相关工作变成新训练授权。远端原评估继续按既定队列；未开始的新研究实验另登记预算。
