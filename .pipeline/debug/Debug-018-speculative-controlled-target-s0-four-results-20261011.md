# 受控T/S0同init四局原生对照结果与下一步

时间：2026-10-11T13:37:07.099251+08:00（Asia/Shanghai）。范围：**投机采样/action-only/LIBERO，不是tile算子**。用户确认前一步建议后实际执行；后台13:16:48启动，2026-10-11T13:34:48.593274+08:00结束，18.00分钟，完成4/4局。

## 1. 本轮回答的问题与证据等级

S0 init4稳定600步的第二罐抓取前有四个无采样双指contact的闭合段，但原先不同init比较不能说明S0相对target额外返工。本轮同init控制：S0先复现旧ON关卡，通过后才T；每条件两个独立EGL worker、每worker首局init3/4、完整horizon800/5稳定步、seed20260922。native与公共指纹都开，teacher/full trace/profile/训练/RL/联训关。公共源码、target、阈值、bonus、cache/scheduler/CFG不改。

- S0完整复现独立CPU复核：True。293/600步与19/38chunk、20初始数组、逐步state/RNG/双RGB/commands/终止、原生物理字段、完整30通道action/video均检查，不仅比较成功与步数。
- T/S0初始已测数组及client RNG/RGB匹配：True；不声称保存/恢复完整server噪声/cache或所有隐藏state。
- 4局单任务开发探索，不是正式成功率非劣或泛化结论；诊断/EGL+指纹有开销，条件顺序S0→T、双worker负载自然变化，不把该wall-clock当clean E2E或同负载成本证明。

## 1a. 按本轮验证顺序整理假设

| 顺序/假设 | 本轮状态 | 证据边界 |
|---|---|---|
| H1 S0在第二罐建立抓取阶段额外返工 | init4有明确对照证据 | S0−T +300步=+8（第一In）+280（第二抓取建立窗口）+12（末段）；280/300=93.33%只是阶段记账，不是因果贡献比例 |
| H2 600步只是target自身同init困难 | 本轮init4不支持 | T同init300步，一段235–298闭合命令已有61双指样本/223.67mm抬升；S0之前四段均无双指、抬升≤6.55mm；不外推其他init/task |
| H3 初始化/观察器造成这次全部差异 | 已测初始化不匹配解释不支持，隐藏因素未穷尽 | 两组20初始数组/client RNG/RGB一致，S0复现旧完整轨迹；前轮ON/OFF已测开关门通过。不证明所有隐藏state/噪声/cache恢复 |
| H4 局部加速被更多工作或高成本批量抵消 | 成本问题仍待验证 | 本轮init4额外19chunk，双worker需求随完成改变；native/指纹开，不把诊断耗时当同负载clean E2E，需要下一成本关卡 |

**目前重点**：不再只问全局L2有没有降，而先检查困难状态下接触前的执行位姿、未有效提起的返工，以及draft实际访问的candidate-state标签覆盖。init3总步数相同（+13/−13/0相互抵消），说明第一处分叉或某阶段变慢并不必然导致整局变长。冻结target、小模型困难阶段同query监督是后续候选，尚未训练；RL/KL或联训不是本轮结论。

## 2. 完整episode结果

| 条件 | init | success | env steps | chunks |
|---|---:|---|---:|---:|
| S0 | 3 | True | 293 | 19 |
| S0 | 4 | True | 600 | 38 |
| T | 3 | True | 293 | 19 |
| T | 4 | True | 300 | 19 |

本轮两对S0−T总步数合计+300；两init并非一致增加；必须逐init解释，不能将init4的600步概括成所有状态都退化。

## 3. 同init阶段分解

| 条件 | init | 第一soup In | tomato首次双指 | success | 第一In→第二双指 | 第二双指→success |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 3 | 168 | 237 | 293 | 69 | 56 |
| S0 | 4 | 167 | 526 | 600 | 359 | 74 |
| T | 3 | 155 | 237 | 293 | 82 | 56 |
| T | 4 | 159 | 238 | 300 | 79 | 62 |

| init | S0−T总步数 | 第一In差 | 接近/建立第二双指窗口差 | 第二双指→success差 |
|---|---:|---:|---:|---:|
| 3 | 0 | 13 | -13 | 0 |
| 4 | 300 | 8 | 280 | 12 |

分解在成功且标记完整的轨迹上是恒等记账，不是因果贡献；第一In是区域包含，不等于释放落稳；首次双指是采样接触标记，不等于稳定抓持。若早期偶然接触或阶段顺序不同，应看完整区间，不以单个标记替代物理事件。

### 第一In之后的闭合命令区间

- **S0 init3**：233–293：57个双指样本，最高抬升229.25mm。第一In之后反退0行。
- **S0 init4**：242–299：0个双指样本，最高抬升4.05mm；342–367：0个双指样本，最高抬升5.47mm；392–425：0个双指样本，最高抬升6.55mm；478–489：0个双指样本，最高抬升0.65mm；523–591：66个双指样本，最高抬升227.79mm。第一In之后反退0行。
- **T init3**：233–289：53个双指样本，最高抬升210.70mm。第一In之后反退0行。
- **T init4**：235–298：61个双指样本，最高抬升223.67mm。第一In之后反退0行。

命令闭合不等于抓到；抬升/EEF-site中心距离/关节slide差仅描述。每env.step之后采样，不能排除内部短接触/力；不证明“z偏高”或L2唯一根因。

## 4. 第一分叉与对照合同

| init | 20初始数组同 | 初始RNG/RGB同 | first command step | first state step | first agentview step | first wrist step | first action chunk | first video chunk |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | True | True | 1 | 1 | 1 | 1 | 0 | 1 |
| 4 | True | True | 1 | 1 | 1 | 1 | 0 | 1 |

完整state字段逐项第一分叉、contacts/objects/goal原生第一分叉、每对应chunk完整action/video逐位计数均保留。T/S0输出差异从第一次请求即可发生时，后面的长阶段只能定位返工，不直接证明哪个去噪深度、动作维度或L2项决定它。

## 5. CPU独立对账与归档

- 95 chunks/1486实际命令按冻结BF16量化动作解码合同重建，逐值匹配step指纹；首chunk只执行12步，终止chunk按done真实裁切。
- 1440保存双RGB观测对与4终止快照hash/垂直翻转合同核对；1490指纹/1490原生行、20初始数组、错误/目标/终止覆盖检查。
- 95对应chunk action/video逐位比较；S0对旧ON全比，T/S0只共同真实chunk，尾部不同长度显式保留，不补假请求。
- 独立CSV/指纹/tensor-summary/input hash复核PASS，676去重文件不变；旧runtime/raw/protected与并行tile数据不覆盖。轮询9次，服务器独立tmux完成；只停止自身pid组。
- 本步没有teacher标签或实际candidate-state训练features，部署action与RGB不能直接当draft监督样本。

## 6. 下一步：按关卡推进，不自动扩展

1. **先读本轮同init的阶段差和失败闭合区间**，区分target自身困难与S0新增返工；不因600步或L2单指标直接决定训练/RL。
2. **另授权并实现/CPU验证同负载T/A_off成本job-only包装**：现有计数依赖active profile与边界CUDA同步，核对真实target工作（含CFG/调用/shape）、request cohort、single-demand完整尾部，不以诊断同步时间当部署收益。GPU诊断最多8局的预算仍未由本轮授权，不能直接启动；profile-off自然E2E另关卡。
3. **训练方案再后置**：继续冻结target，预登记困难阶段同candidate-state teacher标签/覆盖与等预算基线；任务成功、完整环境步数/长尾、实际work/E2E为主，L2辅助。若新状态对照显示增量主要在接近/抓取，应优先验证接触前位姿/抓取返工的任务反馈加权监督；先小规模验证，不直接RL/KL或大小模型联训。真实目标联合训练不能保持当前冻结target的稳定控制，须另立实验。

本轮四局已用尽，不自动加局、成本采集或训练。旧失败根因、正式非劣、泛化与自然E2E收益仍未证明。

## 7. 证据路径

- 原任务：/rjs/cxz/论文探索/投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403
- 原自动报告：/rjs/cxz/论文探索/投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/FINAL_REPORT.md（原件保留，不替换）
- 自动S0 gate：/rjs/cxz/论文探索/投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/s0-reproduction.json
- 自动同init对照：/rjs/cxz/论文探索/投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/matched-target-s0.json
- CPU表：episodes.csv/actions.csv/chunks.csv/observations.csv/native_event_intervals.csv/state_timeseries.csv
- 独立结果：summary.json/pair_comparison.json/stage_summary.json/paired_phase_deltas.json/tensor_comparison.csv/validation.json/input-sha256.json
- 纯数据图：matched_grasp_overview.png；横轴已执行环境步数，图示不充当持稳标签或唯一物理根因。
