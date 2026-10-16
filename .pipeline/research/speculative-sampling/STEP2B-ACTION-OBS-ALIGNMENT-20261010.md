# 投机采样：方法确认与第 2b 步动作/观测对齐

> 后续进度（2026-10-10T23:50:11+08:00）：第 2c 保存数据视觉诊断已定位 init0 搬运掉罐后重搬；物理/模型归因仍未验证，训练未启动。以下保留第 2b 完成时的历史状态；最新见[第 2c 报告](STEP2C-INIT0-VISUAL-REWORK-20261010.md)。

更新：2026-10-10T23:18:28+08:00（Asia/Shanghai）  
范围：action-only 投机采样，与 tile 算子路线隔离。  
状态：**第 2b 步合同内对齐完成；第 2c 步首次停滞/返工诊断待做；训练未启动。**

## 1. 用户确认与研究边界

用户明确表示“可以的，我同意你的这套方法”。记录为方向确认：先冻结真实大模型与 verifier，从现有蒸馏 draft 出发，以组合系统真实任务反馈为候选训练方向；KL 只作定义明确的辅助约束。success、env steps/长尾和完整 E2E 是主指标，真实大小模型权重联训后置。

确认方向不等于 RL 已有效、KL 保证步数，或已经确定 PPO/策略密度/奖励权重/非劣容差/训练预算。仍先完成机制诊断与最小设计；本轮没有模型源码修改、训练、GPU、服务或 LIBERO rollout。

## 2. 本步可见计划和结果

| 子步骤 | 当前状态 | 证据/下一项 |
|---|---|---|
| 2a 数据盘点 | 上轮完成，本轮逐文件扩展验证 | 229 action、217 obs 文件 |
| 2b episode/chunk/执行前缀/观测关联 | 合同内完成 | 12 episode 唯一映射，229 chunk 与计数对账 |
| 2c 首次停滞/返工诊断 | 未完成 | 优先检查 init0，保留其余五对作描述性参照 |
| 3 最小训练对照 | 未启动 | 第 2 步后再明确闭环蒸馏/RL 候选、KL 参考和预算 |

### 2.1 可复查产物

数据入口：[evidence](../../../review/OMISSIONS.md)。

| 文件 | 行数/用途 |
|---|---|
| [episodes.csv](../../../review/OMISSIONS.md) | 12，episode 到输出目录的关联与缺失字段 |
| [chunks.csv](../../../review/OMISSIONS.md) | 229，frame index、实际执行前缀、原生/推导边界 |
| [executed_actions.csv](../../../review/OMISSIONS.md) | 3,507，按源解码合同重建的七维控制命令，不是独立录制的物理状态 |
| [observations.csv](../../../review/OMISSIONS.md) | 3,424，动作后双视角 RGB 的文件、list 下标与图像 SHA-256 |
| [paired_episode_steps.csv](../../../review/OMISSIONS.md) | 6，全部配对 env steps/chunks/E2E，不能删除困难 init |
| [integrity.json](../../../review/OMISSIONS.md) | 数据/合同对账 PASS，不是真实模型数值验证或 RL 结果 |
| [provenance.json](../../../review/OMISSIONS.md) | 485 个输入 SHA-256、源版本、解码函数与 CPU 环境 |
| [execution_contract.json](../../../review/OMISSIONS.md) | 解码函数快照、BF16→F32 运算顺序、首/末前缀和 schema 差异 |
| [validation.json](../../../review/OMISSIONS.md) | 原始 runtime 807 files/376,826,106 bytes 与封存 SHA-256 相同 |

### 2.2 关联方式：不是只按文件名猜

1. 先核对 worker 与合并 episode manifest 的 task/init、起止时间、env steps、chunks、action shapes。
2. 将输出目录秒级时间按 Asia/Shanghai 转为时间点，在同 worker、同 prompt、该 episode 起止窗口内要求唯一候选；用 server reset、policy 与 cache 日志数量再次对账。12/12 映射唯一，未用 chunk 数量单独区分两个同长度 episode。
3. action 下标严格为 0、4、8…；obs 下标覆盖除 terminal chunk 外同一序列。服务器 cache 先保存同 frame index 的执行后 RGB，再推进 index；验证后才建立同下标关联。
4. 最新 init4–5 的 75 条原生 chunk events 均核对 campaign/worker/attempt、连续 chunk、env/simulator counter、done/reward。旧 init0–3 的 154 个 chunk 边界按执行合同＋episode 总步数推导，明确标 inferred；不伪造历史 attempt id、reward 或 done 原生事件。

### 2.3 控制命令恢复与字段检查

- 229 个文件均安全 CPU 加载为 BF16 [1,30,4,4,1]。仅从源文件 AST 提取 CPU postprocess_action，没有导入 WAM/LIBERO 或模型模块。
- 保留 BF16 的标量加法/除法，再由 F32 q01/q99 乘加提升 dtype；复现 epsilon 和七维通道。229 次结果与当前源函数逐值相等；没有先随意转 F32、裁剪或重排动作。
- 按 client 外层 frame、内层 action 的顺序，首 chunk 跳过条件 frame（12 步），之后完整 chunk 16 步，末 chunk 只取成功前执行前缀。合计去除 **48 个条件槽位、109 个未执行尾部槽位**，重建 **3,507 条控制命令**，与 12 个 episode 总步数全部相等。
- 217 个 obs 文件全部使用 weights_only=True 加 narrow NumPy safe_globals；验证 **3,424 组 RGB、6,848 个 uint8 [128,128,3] 图像数组**。没有 unrestricted pickle。
- 每条 RGB 的 env-step 是依已核对执行顺序重建的动作后时刻，不是原生逐图时间戳。12 个 terminal chunk 的 **83 个实际动作后 RGB 未保存**，不是只有最后一张缺失。初始 reset 观测也不由这些 cache 文件补造。
- 命令不等于实际位移/夹爪状态或抓取成功，仍缺物体/eef 位姿、simulator snapshot 和连续 task-progress；本轮没有引入进度估计模型。

### 2.4 历史版本边界

四个 campaign 的 LIBERO config 与当前文件 SHA-256 相同；当前 init4–5 的 server/client 也与封存哈希一致。旧 init0–3 的 server/client 已被后来更新，与当前整文件 hash 不同。旧数据的目录、shape、执行数量、观测数量与日志合同一致，但本步没有找回旧版整文件并验证逐位解码，所以旧命令明确标为基于已审查合同的 reconstruction，不能宣称所有历史实际发送向量都已逐位证明。

launch provenance 同时存在命名键与绝对路径键两种 source_sha256 schema；seed 有数字字符串表示。分析器只规范化读取表示，不改 runtime，也不把 parser 的 schema 差异误报为模型错误。

## 3. 新发现：总体多走步集中在 init0

两臂各 6/6 success；所有六对都保留：

| init | target-only env steps | K2 env steps | K2−target | chunks：target / K2 |
|---|---:|---:|---:|---:|
| 0 | 269 | 441 | +172 | 18 / 28 |
| 1 | 272 | 270 | -2 | 18 / 18 |
| 2 | 265 | 256 | -9 | 17 / 17 |
| 3 | 300 | 290 | -10 | 19 / 19 |
| 4 | 299 | 302 | +3 | 19 / 20 |
| 5 | 270 | 273 | +3 | 18 / 18 |

总步数 1,675→1,832，增加 **157 步（+9.37%）**；其中 init0 是 **269→441，+172 步、+10 chunks**，其他五对合计 **−15 步**。这是对既有配对结果的描述性分解，不是新 rollout，也不是五对排除 init0 后的晋级试验。

它说明“每条 speculative 轨迹都会变长”不符合现有六对数据，提示应优先查困难个例的阶段延迟或返工。**不能据此判断是 L2、夹爪、累积误差、分布偏移或实现问题；不能删除 init0 后宣布任务加速。**六对单任务、全成功也不足以证明成功率非劣或训练可靠 predictor。

## 4. 下一步：仍完成第 2 步，不跳到训练

1. 第 2c 步优先整理 init0 两臂的双视角视觉时间线，按真实累计 env steps 标注粗任务阶段，寻找首次可见的停滞、纠正或返工候选。
2. 将夹爪命令切换、方向反复和动作强度作为描述性辅助；若两臂已访问不同物理状态，不把同一步动作差当成同状态 teacher 监督或反事实因果证据。
3. 保留其他五对参照；缺少真实姿态/成功末段图像的地方明确不可判断。若仍不足，再提出最小完整 episode 的对称诊断采集，遵守 EGL＋独立并行 worker/full horizon 与确认边界。
4. 第 3 步之后才设计闭环蒸馏与任务反馈最小对照。先冻结 target/verifier；KL 参考分布、概率形式、权重、预算与晋级门槛未定，不直接套 PPO，不先联训大模型。

主入口：[RESEARCH.md](RESEARCH.md)；前一轮讨论：[数据与 RL＋KL](STEP2-DATA-AND-RL-KL.md)。每步继续提醒下一步。
