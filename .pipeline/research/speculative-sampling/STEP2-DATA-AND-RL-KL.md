# 投机采样第 2 步：数据盘点与 RL＋KL 候选方案

> 后续进度（2026-10-10T23:18:28+08:00）：第 2b 步合同内动作/观测对齐已完成，首次停滞诊断仍待做；以下保留上轮盘点时的状态。最新见[第 2b 步报告](STEP2B-ACTION-OBS-ALIGNMENT-20261010.md)。

更新：2026-10-10T22:50:59+08:00（Asia/Shanghai）  
范围：action-only speculative sampling；与 tile 算子研究分开。  
状态：**数据盘点子步骤完成，动作/观测对齐与首次停滞诊断尚未完成；RL＋KL 仅是研究假设，没有实现或训练。**

## 1. 本步可见计划

| 子步骤 | 状态 | 下一项 |
|---|---|---|
| 2a 保存数据/字段盘点 | 已完成目录计数、代表性安全 CPU 加载及生成源码核查 | 保留以下字段边界 |
| 2b episode/chunk/执行动作/观测对齐 | 待做 | 验证 worker、输出目录、frame index 和实际执行前缀关联 |
| 2c 首次停滞/返工诊断 | 待做 | 动作与 RGB 的描述性证据；缺少物理状态时不声称接触因果 |
| 3 最小训练对照设计 | 待做 | 第 2 步后比较闭环蒸馏与任务反馈混合训练，RL 是候选而非既定方法 |

本轮未修改模型源码、runtime 原始数据或受保护 state/cycle/rules；未安装依赖、启动服务、训练、GPU 或 LIBERO rollout。只做只读 CPU 盘点、公开原论文查阅及 HW 文档记录。

## 2. 六配对 init 已保存的诊断数据

目录根：[runtime](../../../投机采样/results/runtime)。四个 campaign 均为 libero_10/task0；两臂各覆盖 init0–5。

| campaign | 文件总数 | actions_*.pt | obs_data_*.pt | latents_*.pt | episode 输出目录 |
|---|---:|---:|---:|---:|---:|
| p0-libero-libero_10-task0-init0-3-targetonly-gpuvae-egl-par2-20261006 | 244 | 72 | 68 | 72 | 4 |
| p0-libero-libero_10-task0-init0-3-k2b2-global-gpuvae-egl-par2-20261007 | 274 | 82 | 78 | 82 | 4 |
| p0-phase2b-task0-init4-5-targetonly-gpuvae-egl-par2-20261009 | 143 | 37 | 35 | 37 | 2 |
| p0-phase2b-task0-init4-5-k2b2-global-gpuvae-egl-par2-20261009 | 146 | 38 | 36 | 38 | 2 |
| 合计 | **807** | **229** | **217** | **229** | **12** |

- Target-only 有 109 个 action chunk，K2 有 120 个；两臂各 6 个 episode。不存在这些文件数量等于独立样本数量的关系。
- 四个目录未发现 mp4/avi。已有 RGB 帧可作观察，不必先启动新采集。
- 每个成功 episode 的最后 chunk 不再调用 cache，因此 action 比 obs 文件多 1；12 个终止 chunk 合计造成 12 个文件差额。未保存最后执行后的 RGB，不能用前一帧替代终态。
- init4–5 两臂合计有 75 条原生 chunk event，已有上一轮关联/对账；旧 init0–3 没有该原生字段，必须把按执行合同推导的信息明确标为 inferred，不伪造 reward/done/timestep 原生观测。
- 上述均为目录计数与代表性内容检查，不是对 217 个 obs 内容的逐文件完整验证。

### 2.1 安全加载与真实 schema

使用已有 libero-va 环境、torch 2.9.0+cpu、CUDA_VISIBLE_DEVICES 空、禁用 bytecode。action 的 weights_only=True 加载成功；obs 的初次安全加载被 NumPy globals 拒绝后，只将 numpy.ndarray、numpy.dtype、numpy.core.multiarray._reconstruct 与 uint8 dtype 类型加入显式 safe_globals，仍使用 weights_only=True，没有 unrestricted pickle。

代表性 action 是 BF16 Tensor **[1,30,4,4,1]**。obs_data_0 是长度 12 的 list，obs_data_4 长度 16，每条 dict 只有：

- observation.images.agentview_rgb：uint8 [128,128,3]
- observation.images.eye_in_hand_rgb：uint8 [128,128,3]

代表性目录：
/rjs/cxz/论文探索/投机采样/results/runtime/p0-phase2b-task0-init4-5-targetonly-gpuvae-egl-par2-20261009/workers/worker0/server_output/real/put both the alphabet soup and the tomato sauce in the basket_20261009_153223/

源端 [_extract_obs](../../../投机采样/scripts/p0_libero_multichunk_client.py)（129 行起）也只保存两路 RGB；没有 eef pose、真实夹爪状态、物体位姿、simulator snapshot 或连续 task-progress 字段。cache 请求的 state 实际是预测 action，**不是 simulator state**。chunk event 保存 shape、计数、reward 摘要和 done，但不保存逐步 action 数值或物理状态。

### 2.2 不能直接拿文件做动作比较

[server](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)（2603–2608 行）在 postprocess_action 之前保存 actions。因此文件是模型归一化输出，不是实际送入 env.step 的七维动作。

恢复时必须忠实复现 [postprocess_action](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)（672–682 行）和 [LIBERO 配置](../../../投机采样/wam-p0-worktree/wan_va/configs/va_libero_cfg.py) 中的 q01/q99、epsilon、通道 0–6 与 dtype 路径，不做任意裁剪；恢复结果应是 [7,4,4]。

[client](../../../投机采样/scripts/p0_libero_multichunk_client.py)（391–413、438–466 行）首 chunk 跳过条件 frame，通常执行 12 个 env steps；之后完整 chunk 通常 16 steps；成功终止 chunk 只执行前缀。不能把固定条件帧或未执行尾部纳入运动/夹爪统计。

文件下标 0、4、8… 是服务器 frame index，不是 env step。obs_data 来自执行后 cache 请求（server 2615 行起），action 与 RGB 要按实际生成时序关联。episode manifest 没有 server_output_dir，必须先验证 worker＋reset/output 时间＋目录顺序映射，不能随意按文件名连接；原生 trace/event 还要检查 episode_attempt_id。

同 task/init 也不代表两臂每一 env step 的物理状态相同。累计步数对齐或视觉阶段对齐只能作描述性比较，不能把跨状态逐点动作差异当成同状态 target 标签或反事实因果结论。

## 3. RL 是否可能有效：值得检验，但不是既有结论

RL 可以把优化方向从“像 teacher”移到组合系统的真实任务结果，因而比只压平均 L2 更贴近目标。扩散机器人策略的任务反馈微调已有 [DPPO 原论文](https://arxiv.org/abs/2409.00588) 的方法证据；它不直接证明当前 LingBot-VA flow draft＋离散 verifier/rollback 会更快、走更少步或可直接套 PPO。

先冻结真实 target、verifier、scheduler 和执行协议，从现有蒸馏 draft warm start。环境中评测与奖励对象是 **draft＋真实 target/verifier＋accept/rollback**，不是独立小模型。被拒绝 proposal 的训练信用分配、稀疏任务奖励和采样/似然定义仍需解决。真实 WAM 大小模型权重联训保留为更后面的独立实验。

## 4. KL 究竟约束什么

这里先按用户上下文把 trace 长度解释为 **LIBERO 完整 episode 的 env steps**。去噪 scheduler updates、speculative accepted-prefix 长度和 env trajectory 长度必须分开报告。

### 4.1 普通常用 policy KL 不直接约束步数

[TRPO 原论文](https://proceedings.mlr.press/v37/schulman15.html) 的 trust region 约束旧/新策略的分布变化；实际平均 KL 近似也不是环境轨迹长度或成功率的硬保证。局部小更新经过多步累积仍可能改变抓取成败或造成返工。

参考对象必须写清：

- 对旧 draft 的 KL：限制单次更新，不防止多轮逐渐远离原始策略。
- 对冻结初始 draft 的 KL：限制偏离蒸馏基线，但过强会阻碍有用改进。
- 对真实 target 的 KL：必须有可比较的概率分布/密度；当前 teacher 连续方向不能直接当概率。

若指 env-step 长度分布 P(N) 与基线 Q(N) 的 KL，它更直接涉及长度，但不保证成功，也可能惩罚更短的好轨迹；当前六对样本不足以可靠估计整个分布。优先直接记录配对步数、预算和长尾。

### 4.2 当前实现没有现成的 PPO/KL 似然

[TwoHeadResidualActionPredictor.predict](../../../投机采样/scripts/p0_batched_verify.py)（303–352 行）给定条件返回两个连续方向 [2,*current_state.shape]；没有 head 概率、方差或 log_prob。两个 head 是 proposal，不是已经归一化的分类分布。初始噪声仍可造成隐式动作随机性，不能说整个系统没有随机性，但当前接口没有可直接计算的条件密度。

不能对任意动作坐标做 softmax 来制造有语义的 KL。确定性不同点输出若视作 Dirac 分布，普通 KL 通常不有限。

若仅假设两边都是相同固定协方差的 Gaussian，则：

    KL(N(mu_d, Sigma) || N(mu_t, Sigma))
      = 0.5 * (mu_d - mu_t)^T * Sigma^-1 * (mu_d - mu_t)

这就是加权平方 L2/Mahalanobis；将 L2 改名为 KL 并未解决 task-progress 与步数问题。PPO 风格方案需先设计随机 proposal、合法密度及 old/new likelihood 记录；对 teacher 的监督可以保留为辅助约束，但不冒充 teacher 分布 KL。本轮未授权或实施这些模型更改。

### 4.3 全轨迹 KL 的数学边界（本记录的推导，不是实验）

若同初始分布、同环境动力学、有限共同 horizon H、终止后吸收状态填充，并且真正控制整条执行轨迹分布 P/Q 的 KL，N∈[0,H] 时由 TV/Pinsker 可得：

    |E_P[N] - E_Q[N]| <= H * TV(P,Q) <= H * sqrt(KL(P||Q)/2)

它只是这些强假设下的平均长度上界；不是逐 episode 或长尾保证，通常很松。去噪方向的逐点 KL、old/new 的平均 KL 或两条保存样本不等于这样的全轨迹 KL 约束。

## 5. 候选训练目标与评测合同

优先级：**success 非劣 → env steps 相近/更少且长尾不过度增加 → 完整 E2E 更快**。

可讨论的任务反馈目标（未定权重/算法）：

    task success / 可验证进度
      - alpha * 实际环境步数
      - beta * 组合系统真实推理成本
      - lambda * 定义明确的 KL 正则

这只是候选形式，不是“奖励写出来就有保证”。success 非劣、配对 step budget 与长尾应另设晋级门槛，容差和失败计费规则须预先确定。提前失败不能因为步数小获益；报告所有 episode，并单列双方成功的配对步数。未验证的连续 progress 不直接加入奖励。

实际推理成本应包含 rejected proposal、verifier、fallback、cache 和 reset 等系统账本，而非只奖励接受长度或去噪步数；训练用噪声/探索和部署确定性设定也须单列。KL 可以辅助稳定训练，不能代替任务门槛；不同时修改 verifier gate 以免混淆归因。

## 6. 下一步：先完成第 2 步，再谈训练启动

1. 先在已有 init4–5 原生 event 配对上，核对 episode→服务器输出目录→frame index→chunk 的关联及首/末执行前缀，再扩到旧 init0–3；不启动新环境。
2. 忠实反归一化七维动作，仅统计实际执行部分；观察夹爪命令切换、方向反复和动作强度。命令不是实际抓取/夹爪状态。
3. 双视角 RGB 按实际 env-step 时间和粗任务阶段整理，定位首次可见停滞或返工候选；没有物体位姿或最终 obs 的部分明确标缺失，不声称因果已确定。
4. 若上述证据仍不足，提出最小完整 episode 的对称诊断采集方案，遵守 EGL＋并行 worker、全 horizon 与确认边界；不做 smoke/截断替代。
5. 第 3 步再确定闭环蒸馏基线、任务反馈训练、KL 参考和可计算概率形式；需批准模型更改/训练后才执行。每次输出继续提醒下一步。

主入口：[RESEARCH.md](RESEARCH.md)。本记录不改写历史代理/闭环实验为成功，也不推进 tile 算子里程碑。

## 7. 写入后校验

116 个既有受保护/源码/记录路径保持原 hash 或原缺失状态；四个 runtime campaign 的 807 files、376,826,106 bytes 逐文件 SHA-256 与上一轮封存一致；旧索引条目、compact 行与 PROGRESS/log 前缀保留（包括独立 tile 记录）。新 Knowledge record schema 与 secret gate 通过，文档本地链接有效。上述为记录/原始数据完整性校验，不是 RL 或 LIBERO 结果。

校验：[STEP2-VALIDATION-20261010.json](STEP2-VALIDATION-20261010.json)。下一步仍是第 2 步动作/观测对齐及首次停滞诊断。
