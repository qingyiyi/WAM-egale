# 投机采样固定负载成本：GPU结果、CPU终审与下一步

2026-10-11T23:07:03+08:00（Asia/Shanghai）。仅投机采样主线，**非tile算子研究**。

## 1. 当前状态：本次已授权任务完成

本次 job 已完成真实 GPU 启动、初态/缓存恢复、不扰动检查、四格成本与清理；终审通过。**PASS 指实验有效性，不代表四格均加速，也不代表闭环质量非劣。**

- [运行目录](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036)
- [批准合同](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/plan.json)
- [实际最终结果](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/FINAL_RESULT.json)
- [CPU事后复核](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-213036/CPU-final-review/AUDIT.json)

| 项目 | 实际结果 / 边界 |
|---|---|
| 新完整 episode | 0；没有执行 policy 环境动作 |
| EGL 初态重建 | 2/2，init3、init4；各5步零动作稳定，共10环境稳定步，不是完整任务评测 |
| 固定完整 chunk 调用 | 168/168，112个请求槽；参考/包装预热48次、ON诊断24次、正式OFF主测96次，含各自双需求peer |
| 原始缓存引导 | 每worker 14个原prefix chunk，共84次实际Transformer前向；单独记账，不冒充168次完整调用 |
| GPU安全计时 | 21:45:37启动，22:54:15结束；4118.325秒，约68分38秒，低于90分钟硬限时 |
| 训练 / teacher / RL / 联训 | 均未启动；target、A_off、阈值、bonus、CFG及公共源码不改 |
| 检查关卡 | INITIAL、FIXTURES、G1、S1、L2、L1、S2、FINAL全部通过；AUTH/CPU/START与各GPU关共11项HW记录已写入 |
| 正常清理 | OWNERSHIP-CLEANUP通过；已知自有模型、EGL、监督器PID不存在，自有5个端口不监听 |

此前预加载生命周期错误和私有监控归属误报已修复并分别保存失败记录。**本次实际原reset→A_off预加载成功、两GPU rank未被误报、完整结果已取得**，不是仅CPU模拟通过。CPU事前221项合并回归与26项私有归属回归通过；事后复核重新核对1380份冻结文件SHA、9份target权重size/mtime，均未改变，protected state/rules一致，cycle缺失状态保持。权重检查不是完整target权重SHA；测试、实现与终审由同一agent完成，不称独立第三方审计。

私有归属监控正常随任务退出，273次采样未见外来可见compute进程，未向外来进程发送信号；离散采样不能证明采样间隙或graphics任务完全不存在。本次后台服务器tmux完成了实际任务；其独立于本机SSH/熄屏，但不能保证服务器重启后继续。

## 2. 计时与可比性：实际GPU路径，不是CPU模型性能

真实 target/VAE 与 A_off 在A100 CUDA执行，文本编码保持原CPU配置。动作计时采用host clock和CUDA边界同步，包含host dispatch，**不是CPU模型计时，也不是纯GPU kernel time**。

正式结果使用诊断 **OFF** 的 action分支中位数（包含末尾terminal forward）。每个fixture/算法/需求有4次primary重复，ABBA/BAAB两顺序块。双需求是同卡两真实模型并发，peer不是独立episode；主测动作窗口重叠检查通过。诊断ON单独保存、不混入主测。计时边界外的cache/artifact核验与等待可能增加总job时间；人工同步门等待不充当自然任务E2E。

同算法 reference / wrapper / observer ON-OFF 的全部物理action、video、30层有效KV及元数据、RNG与工作计数精确比较通过。**这不等于 T 与 A_off 之间动作完全相同**；也不等于ON计时不受观察器影响。

短fixture对应action有效KV288，长fixture对应KV2160。它们来自不同真实历史状态，单双需求内分别恢复同一fixture。短/长比较同时改变了状态和接受路径，不能把全部差异唯一归因为KV长度。

## 3. 正式OFF成本矩阵

以下均为**动作分支秒数**，不是完整chunk或完整episode E2E。百分比为 A_off/T−1，负数更快。

| 历史 / 同卡需求 | init | T 中位数(s) | A_off 中位数(s) | A_off/T | 耗时变化 |
|---|---:|---:|---:|---:|---:|
| S1 短 / 单 | 3 | 4.4838 | 3.0790 | 0.6867 | −31.33% |
| S1 短 / 单 | 4 | 4.3229 | 2.9858 | 0.6907 | −30.93% |
| S2 短 / 双 | 3 | 7.5022 | 6.3867 | 0.8513 | −14.87% |
| S2 短 / 双 | 4 | 7.2715 | 6.1604 | 0.8472 | −15.28% |
| L1 长 / 单 | 3 | 4.7442 | 4.9651 | 1.0466 | +4.66% |
| L1 长 / 单 | 4 | 4.5754 | 5.0867 | 1.1117 | +11.17% |
| L2 长 / 双 | 3 | 8.1618 | 10.6287 | 1.3022 | +30.22% |
| L2 长 / 双 | 4 | 8.2988 | 10.6400 | 1.2821 | +28.21% |

- S1 / S2 均为 `speedup_pilot`，两顺序块方向一致。
- L1 预注册5% material screen下为 `flat_or_mixed`：init3中位数略低于5%慢化阈值，init4明显慢化；**不能把mixed写成没有代价，更不能写成加速**。两个init的两顺序块均未显示净收益。
- L2 为 `cost_regression_pilot`，两init及顺序块均慢化。慢结果属于有效结果，不触发重试或删样本。
- 所有主测MAD/median低于7%门限，观察最大约3.45%；小pilot不是统计显著性、跨任务推广或质量非劣证明。

同一fixture/算法单双需求的真实工作量保持相同。双/单耗时比：T约1.67–1.81，A_off约2.06–2.14。**本设置下 A_off 对同卡并发更敏感**；没有据此声称独占自然单worker也会同样退化。

## 4. 工作账本：减少调用，不等于减少候选计算

每个固定chunk均完成原50次action scheduler更新及1次terminal cache forward。下表统计action target实际前向与累计CFG行调用量；CFG行调用量是工作规模指标，**不是精确FLOPs或时间单位**。

| fixture | T 前向 / CFG行调用 | A_off 前向 / CFG行调用 | verifier树次 | draft前向 |
|---|---:|---:|---:|---:|
| 短 init3 | 51 / 102 | 30 / 384 | 27 | 81 |
| 短 init4 | 51 / 102 | 29 / 370 | 26 | 78 |
| 长 init3 | 51 / 102 | 32 / 412 | 29 | 87 |
| 长 init4 | 51 / 102 | 32 / 412 | 29 | 87 |

K2B2 verifier每树展开7个逻辑候选，即14个CFG行。前向数量下降，但累计CFG行调用达到原来的约3.6–4.0倍；相同fixture/算法跨单/双需求与重复的工作计数一致。接受深度0/1/2次数：短init3为6/6/15、短init4为4/7/15、长init3为10/6/13、长init4为10/7/12。当前bonus关闭；接受深度是内部去噪更新，不是可执行环境动作长度。

这使“先提高接受率就一定更快”成为需要成本验证的假设，而不是训练依据。长历史下树更频繁、行量更高与并发更敏感同时出现，但本轮不能隔离每个因素的因果贡献。下一步应审查候选规模、验证工作复用和成本准入，再决定训练的收益上限；**不是直接放宽L2、改bonus或更换冻结target**。

## 5. 诊断ON只用于瓶颈线索

各条件只有1次primary ON诊断，含peer共24个诊断调用。A_off中 `verifier_target_forward` 占ON action约69.3%–83.7%，`draft_model`约2.2%–4.7%。由此提出**优先检查verifier预算**的线索，但不称精确OFF kernel归因。

保留ON异常：S2 T init4的ON action约23.6416秒，而其正式OFF中位数为7.2715秒；S1 T init3的ON action约6.8212秒，而OFF中位数4.4838秒。G1数值/工作不扰动不保证观察器时间不扰动。这些单次值不用于裁定算法加速、不能静默剔除，也不把OFF−ON当精确诊断开销。`action_tree_build`包含draft/scheduler等嵌套phase，禁止重加重复计数。

## 6. 研究范围与CCF A：不重构，但扩大可迁移性验证

完整原始来源与先行工作边界见 [FLASH/Fast-WAM文献审查](REALTIME-VLA-FLASH-FASTWAM-SCOPE-20261011.md)。该文献报告第6节的running是当时快照，不改写；本报告与RESEARCH最新头给出完成状态。

需要区分物理动作horizon H、内部去噪步数 S、并行验证候选 B。当前LingBot已整块产生16未来action，但内部50次更新；[Fast-WAM](https://arxiv.org/html/2603.16666v2)视觉single-pass而action仍10步，不能理解为一次forward完成整个动作求解。[π0/0.5官方实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)默认10步；这类轻动作专家更能检验方法的现实收益边界。

[Realtime-VLA FLASH](https://arxiv.org/html/2605.13778v1)已对π0整块动作进行draft/并行端点验证/可执行前缀及阶段回退；本项目draft去噪transition，验证对象不同，但不能宣称首个连续动作投机，也不能把简单阶段fallback作为独立新意。[ICML2025连续扩散投机](https://proceedings.mlr.press/v267/de-bortoli25a.html)进一步说明向量连续输出不排斥投机；其精确随机采样保证不自动转移到本项目flow+L2验收。

**建议，而非已批准的新实现：**保留LingBot作为长历史压力案例，以Fast-WAM或π类5–10步动作专家作为后续迁移候选。研究贡献收束到“任务成功非劣、环境步数/失败长尾受控、且在真实KV/需求成本下有净收益的投机推理”；不是为了模型名称换成‘主流’，也不是声称现有pilot足以发表CCF A。Fast-WAM是代表性候选，未统计其使用占比，也未核实其CCF A录用状态。必须对照已优化原模型、质量匹配少步解算及适用的FLASH式方法；算法与kernel/compile收益分开。

## 7. 未完成项和下一步（无追加GPU或训练）

本轮尚未证明：成功率非劣、环境步数保持、完整episode E2E改善、KV单因素因果关系、跨模型/跨任务有效性。原先的任务返工/环境步数目标仍未解决；固定成本完成只是必要前置，不是最终目标完成。

建议下一步仅CPU制定两个串行关卡，完成一关后继续提醒下一关：

1. **成本可行性关**：用已保存账本与真实K2B2循环核对最优接受下的tree/CFG行量边界，列出compact verifier、工作复用、负载保守启停的候选方案和可证范围。区分行量界与wall-time界，不用当前计数直接许诺加速；本轮未给出最佳接受行量推导。
2. **可迁移性/新意关**：先核实一个现代Fast-WAM/π类公开checkpoint、真实接口、5–10步设置、优化基线及FLASH可比范围；为未来配对闭环评测预注册success容差、环境步数/长尾、失败处理、E2E与样本预算。正式非劣容差尚未定。

这两关通过后才建议单独确认最小实现与新GPU预算；任何缩树、启停、换target、自然闭环、训练/RL/KL/联训仍需另确认。暂不以高接受率直接启动训练，也不混入tile成果。
