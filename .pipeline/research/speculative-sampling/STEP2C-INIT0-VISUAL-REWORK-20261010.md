# 投机采样第 2c 步：init0 掉罐与重搬视觉诊断

> 语义补充与后续设计（2026-10-11T00:09:07+08:00）：本任务 basket site 的 check_contact 恒真，In 不能当真实接触或稳定抓持；原报告以下保留当时的通用代码说明。最新诊断/最小对照设计见[第 3a 报告](STEP3A-DIAGNOSTIC-AND-MINIMAL-COMPARISON-20261011.md)，observer/采集/训练未启动。

更新：2026-10-10T23:50:11+08:00（Asia/Shanghai）  
范围：第一代 LingBot-VA action-only 投机采样；与 tile 算子研究隔离。  
状态：**保存数据的视觉返工定位完成；物理/模型根因和原生进度验证未完成；训练未启动。**

## 1. 本步计划与完成程度

| 子步骤 | 状态 | 结果 / 下一项 |
|---|---|---|
| init0 两臂全程概览 | 完成 | 非终止 chunk 末帧双视角，定位首次搬运可疑区间 |
| 首次异常逐帧检查 | 完成 | 84–108，重点 98–104；约 101–103 出现掉罐 |
| 命令 / 视觉 / chunk 对账 | 完成（合同重建边界保留） | 掉落处 chunk 6、server frame 24、env steps 93–108；仍发闭合命令 |
| 其余五对命令段检查 | 完成，仅数值 | 全部各两段正夹爪命令；不代表完整视觉检查或无掉落 |
| 真实抓持 / 接触 / 入篮状态验证 | 未完成 | 旧数据没有这些值；源码候选不等于运行验证 |
| 最小冻结 target 对照设计 | 下一步 | 先规定真实诊断字段、比较条件和判据，不直接训练 |

本輪仅 CPU 安全读取既有缓存与源码、绘制科学图板、记录 HW；未修改模型源码、启动 GPU/服务/LIBERO rollout/训练或安装依赖。旧工程 state completed 不代表本研究完成。

## 2. 核心新证据：第一次搬运掉罐，之后重搬

六对旧结果中，init0 为 target-only 269 步、K2 441 步（+172）；总增量为 +157，其他五对合计 −15。保留全部配对，不排除困难 init 后声称晋级。

人工观察对象别名 A=蓝黄罐，B=绿红罐；按外观与动作连续性识别，**没有 simulator object-id 验证**。

| env-step（动作后图像） | K2 可见过程 | target-only 参照 / 边界 |
|---|---|---|
| 84–99 | A 仍紧邻手部被搬运 | target 也可见相应搬运；同计数不等于同物理状态 |
| 100 | A 仍靠近手部，是否已滑移不确定 | target A 仍在手部附近 |
| 101 | A 相对手部明显下移，可能仍有单侧接触 | 不能由图像恢复接触力 |
| 102 | A 与手部视觉上明显分离 | 不是原生失去接触事件时间戳 |
| 103–108 | A 回到桌面位置，手部空着继续移动 | target 对应窗口中仍搬运 A，108 可见仍随手部 |
| 124 | 空手到篮子上方，A 仍留桌面 | 支持“此次搬运中断”，不是先入篮再掉出 |
| 148（target） | — | 腕视角 A 在篮子开口区域；原生 In 真值缺失 |
| 236（target） | — | target 已搬运另一绿红罐 B |
| 268（K2） | 再次搬运蓝黄罐 A | 具体返工证据；相同外观对象的人工识别 |
| 316（K2） | 重搬后 A 在篮子区域 | 视觉入篮不等于原生成功谓词 |
| 396（K2） | 之后才搬运绿红罐 B | 与第三段闭合命令对应；命令段不等于成功抓取次数 |

![首次搬运掉罐：左 target，右 K2；上 step100，下 step103](/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/step2c-init0-visual-rework-20261010/first_drop_compact.png)

图像来自原始 128×128 RGB，明确采用最近邻 2× 放大；没有图像合成或插值重建。逐帧图板见 [98–104](../../../review/OMISSIONS.md)、[84–108](../../../review/OMISSIONS.md)，后续见 [K2 重搬](../../../review/OMISSIONS.md)。

**可以判断**：K2 此轨迹存在第一次搬运掉罐、随后重搬的可见返工过程，是环境步数增加的具体解释线索。  
**仍不能判断**：真实接触/滑移最早发生时间、掉落的物理或去噪根因、是否由某条 accepted edge 造成，以及全部 +172 步有多少由此事件导致。首次可见任务失败不是首次潜在控制误差。

## 3. 不是简单的“夹爪命令提前松开”

client 默认 Panda / OSC_POSE / default gripper 的源码链，以及 [PandaGripper.format_action](../../../review/OMISSIONS.md) 显示：负命令为打开、正命令为闭合；执行使用 sign。这里仅作源码语义核查，没有读取真实夹爪关节或力。

| init0 | 正 command_6 连续区间（闭合命令） | 第一次后续负命令 |
|---|---|---:|
| target-only | 60–138；210–262 | 139 |
| K2 | 60–135；239–298；377–434 | 136 |

两臂都在第 60 步开始第一段闭合命令。**K2 约 101–103 步掉落时仍是正闭合命令**；98–104 两臂 command_6 全为正，K2 到 136 才转负。因此，之前仅按较早“松手”切换猜测首次放置失败，不能成立为该掉落过程的解释。实际夹爪开度、碰撞、接触力与持稳状况仍未知。

全 12 episode 的数值检查：K2 init0 有三段正夹爪命令，其他 11 个 episode 均有两段。此结果与 init0 多一次搬运的视觉过程一致，**但三段正命令不能自动计为三次成功抓取，也不能证明其他五对无掉落/返工**。

[命令时间线](../../../review/OMISSIONS.md) 中的 0:3 和 3:6 范数只是重建命令描述量，不是测得的机器人速度、位姿或接触状态。两臂已到不同物理状态时，同一步的命令差不构成同状态 teacher 标签或反事实因果证据。

## 4. 目前支持、削弱与仍开放的假设

| 假设 / 解释 | 本轮结论 | 还需什么 |
|---|---|---|
| 返工使轨迹变长 | 有具体过程证据：掉罐后重搬 | 原生阶段/持稳事件、更多任务和配对；不能量化归因全部 +172 |
| 简单提前打开夹爪 | 被时间顺序削弱：掉落时命令仍闭合 | 实际 gripper qpos/contact，区分命令与执行状态 |
| 最初抓取不稳 / 搬运姿态或接触差异 | 仍是候选，尚未验证 | 抓取前起的 eef/object 相对位姿、双指接触、真实关节状态 |
| 失败抓取后的恢复或线上分布偏移 | 仍是候选；可见空手继续到篮子、之后重搬 | 同实际访问状态 teacher 标签和系统 cache/noise/验收 provenance |
| 平均 L2 导致掉落 | 未建立因果联系 | 同状态合同下的可追溯局部误差与物理响应；不能跨轨迹同计数硬配 |

此结果把“弱化 L2 的任务代理地位”落实到**真实抓持、搬运连续性与返工事件**，不意味着放宽 verifier 阈值，也不证明某个任务损失或 RL 已有效。

## 5. 原生任务进度候选：两项入篮谓词，不虚构连续距离

[本任务 BDDL](../../../review/OMISSIONS.md) 的目标为两个 In 的 conjunction：

- In alphabet_soup_1 basket_1_contain_region
- In tomato_sauce_1 basket_1_contain_region

真正的 [living-room task class](../../../review/OMISSIONS.md) 在 _check_success 中逐项 _eval_predicate；[In 实现](../../../review/OMISSIONS.md) 要求目标区域 contact 且 contain。因此可将**两物体各自的原生目标布尔值**列为 task-progress 候选，而不是凭图像推定成功、发明连续 goal-distance，或使用抽象基类返回 False 的默认 _check_success。

[robosuite _check_grasp](../../../review/OMISSIONS.md) 的默认源码检查两侧 fingerpad 与物体的接触，可作持稳诊断的一个候选；单一接触布尔值不证明稳定抓持。

这些都是**源码级候选**：运行时绑定、逐步读取、成本与正确性未测；旧 RGB 没有对应真值，不能回填训练标签。实际 gripper qpos、eef/object 相对位姿、接触时序及两项目标值须在后续最小诊断设计中核实。特权状态只作诊断/训练辅助，不能悄悄成为 simulator-only 的部署输入。

## 6. 证据与验证

入口：[evidence](../../../review/OMISSIONS.md)。

| 产物 | 规模 / 用途 |
|---|---|
| [manual_visual_events.csv](../../../review/OMISSIONS.md) | 13 条人工视觉注释，含 obs 文件、下标、两视角 hash；非物理真值 |
| [positive_command6_intervals.csv](../../../review/OMISSIONS.md) | 全 12 episode 共 25 段正命令 |
| [episode_command_summary.csv](../../../review/OMISSIONS.md) | 12 个 episode 命令摘要 |
| [first_transfer_commands_84_108.csv](../../../review/OMISSIONS.md) | 两臂共 50 条合同重建控制命令 |
| [init0_chunk_command_features.csv](../../../review/OMISSIONS.md) | 46 chunk 的描述量；不是独立 episode 样本 |
| [visual_provenance.json](../../../review/OMISSIONS.md) | 44 原始 obs hash、局部源码/输入与临时 CPU 分析方法来源 |
| [candidate-signals.json](../../../review/OMISSIONS.md) | 原生进度/抓持候选与运行验证缺口；不是已收集数据 |
| [validation.json](../../../review/OMISSIONS.md) | 输入、图像、命令段、原始 runtime 不变及 HW 增量保留检查 |

44 个 obs 文件安全重新加载（weights_only=True，窄 NumPy 白名单、CPU）；696 组动作后双视角、1392 图像数组与第 2b 索引 hash 相同。target 保存 268 帧（terminal 第269步缺失），K2 保存 428 帧（terminal 429–441 缺失），初始 reset RGB 未存。因此不能逐帧确认最后的成功状态。

所有 env-step 映射沿用第 2b 的已审查执行合同。旧 init0–3 source 整文件与当前不同，历史实际发送向量逐位等价仍未证明；本报告用**重建控制命令**措辞。807 个原始 runtime 文件（376,826,106 bytes）、旧 records、源码与第 2b evidence hash 不变。受保护 state/rules 未改，cycle 未创建；tile 记录和旧索引保留。

## 7. 下一步：最小对照设计，不直接训练

1. **先定诊断采集合同与可行性检查**：逐步真实动作及时间关联、actual gripper qpos、eef/object 相对位姿、双指接触、两项原生 In；终止时刻也记录。必须区分真值、源码候选与派生事件，明确缺失项与成本。诊断 run 与干净 timing 分开；不假定 snapshot/同状态反事实恢复已可用。
2. **冻结真实 target/verifier 与执行合同**，设计最小 A（现有离线 draft）、B（系统访问状态上的 teacher 监督）、C（加入任务反馈的混合候选）对照，先选一个最小问题，不同时改变 verifier gate、target 权重和 draft 后混淆原因。优先判断“闭环关键持稳/恢复阶段”是否值得针对性训练；没有证据说明只加强夹爪二值监督就足够。
3. **规定评价与预算**：success 非劣容差、完整 env steps、失败/长尾处理、E2E、held-out task/init 和 teacher/交互/训练成本；数值尚待约定。继续保留 init0，不用六对或手工图像标签训练可靠 predictor。
4. **然后再选训练算法**：KL 是辅助，不保证 trace 长度；当前确定性 draft 缺密度/log_prob，固定同协方差 Gaussian KL 又退化为加权 L2。必须先说明随机策略/参考、accept/rollback 信用分配与稀疏任务反馈如何用，再决定 RL；大模型真实权重联训后置。

模型/source hybrid 修改、新采集、训练或服务/GPU 操作须明确启动确认。未来真正采集按项目 EGL＋独立并行 worker/full horizon；单 A100 上不同算法条件串行。**本轮只到保存数据诊断与候选整理，不执行这些后续步骤。**

持续入口：[RESEARCH.md](RESEARCH.md)；前一轮：[动作/观测合同对齐](STEP2B-ACTION-OBS-ALIGNMENT-20261010.md)。每次输出继续提醒下一步。
