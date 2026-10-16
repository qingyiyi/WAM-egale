# 投机采样第 3a 步：真实诊断合同与最小冻结 target 对照

更新：2026-10-11T00:09:07+08:00（Asia/Shanghai）  
范围：第一代 LingBot-VA action-only；HW 与 tile 算子路线隔离。  
状态：**设计与源码可行性核查完成；15 项窄源码 CPU 语义检查通过。观察器未实现、真实字段未采集、训练未启动。**

## 1. 本步完成程度与下一步

| 工作 | 状态 | 不能推出什么 |
|---|---|---|
| 真实姿态/夹爪/接触/入篮 provider 核查 | 静态与窄 CPU 语义核查完成 | 运行时绑定、成本和物理根因已验证 |
| 默认关闭观察器的数据/terminal/不扰动合同 | 设计完成，未实施 | 日志已正确采集 |
| 最小分阶段对照与评价合同 | 候选设计完成，预算/容差/划分未定 | 已批准四episode或训练执行 |
| client 观察器实现与 CPU 合同测试 | **下一步待确认源码范围** | 本轮用户同意设计已授权修改源码 |
| 真实环境采集 / 训练 / 大小模型联训 | 未开展 | 静态 15 tests 是 LIBERO 成功率证据 |

本轮采用轻量 HW research/chat 增量记录，不创建新 Cycle/Milestone，不修改受保护 state/cycle/rules。旧工程 completed 不是投机研究完成。无模型/source 修改、GPU、服务、rollout、依赖安装或训练。

## 2. 核查中发现的关键语义与风险

### 2.1 入篮谓词不是物理接触/抓持信号

[对象 wrapper 工厂](../../../review/OMISSIONS.md) 将 object_sites_dict 中的区域包装为 SiteObjectState；[SiteObjectState.check_contact](../../../review/OMISSIONS.md) 恒返回 True。本任务 In 调用区域 contact 和 contain，所以**这里的原生 In 实质主要检查区域包含**，不是篮子和罐的接触力、双指接触或稳定放置证明。

第 2c 报告中“In 为 contact AND contain”是通用代码形式；本轮补充实际 site contact 的恒真语义。两个原生目标布尔值仍是任务定义的有效候选，但不能据此创造真实抓持标签。实际运行需核实区域实例类型与两个 goal flags 和 [真实任务 _check_success](../../../review/OMISSIONS.md) 的合取一致。

### 2.2 姿态必须拷贝，坐标和四元数必须注明

[ObjectState.get_geom_state](../../../review/OMISSIONS.md) 直接返回 sim.data.body_xpos/body_xquat 的数组视图；不 copy 就缓存，后续步可能覆盖旧记录。CPU 合成数组检查证明该视图会随底层数组修改。

对象 body 四元数是原生 wxyz，而 [robot eef observable](../../../review/OMISSIONS.md) 明确转为 xyzw；site mat2quat 也为 xyzw。诊断记录保留 provider/frame/order，并明确转换。eef 的位置和姿态统一用 grip site 的 world position 与 rotation matrix，避免将 grip-site 位置和 raw-observation 中 eef body 四元数悄悄拼成同一实体位姿。

真实夹爪 qpos 按 joint 名/地址读，不能用 command_6 的正负或大小代替夹爪开度。左右 fingerpad 对每个物体的接触分别记录；[默认 _check_grasp](../../../review/OMISSIONS.md) 为双侧接触合取，仍不是持稳或力。接触力读取、单位和额外成本本步未验证，不列为首版必需。

### 2.3 必须记录 terminal，不能改变模型输入

[当前 client](../../../投机采样/scripts/p0_libero_multichunk_client.py) 在 env.step 后先检查 done 并 break，才提取后续 RGB；这正是旧 terminal 图像缺失的路径。拟议 observer 必须在这个 break **之前**记录实际动作后状态，并在现有五步稳定化结束后记 env-step 0 的初始状态。

特权状态只进独立诊断文件；现有模型仍接收原来的 RGB。不得为采集额外 env.step、sim.forward、reset、_post_process、observable refresh、infer/cache 或改 RNG/动作/成功/执行前缀。没有完整 simulator+cache+RNG restore 证据，不承诺同状态模型反事实分支可用。

### 2.4 现有紧凑 teacher 日志不等于线上蒸馏数据

[trace record_scheduler_update](../../../投机采样/wam-p0-worktree/wan_va/utils/p0_trace.py) 在 compact draft_train 模式跳过 model_source=draft，保留 target action commit；CPU 提取方法已验证这个分支。仅采它会漏掉已接受 draft 更新对应的状态，或偏向 target fallback。

[server K2 trace 绑定](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py) 和 full tree 记录包含真实 verifier query 的 candidate state 与 target direction，可以作为线上标签来源。但 compact/stats 的 tree 仅有摘要；full tree 本身也不自动包含完整 predictor 特征，尤其 root 的 previous_direction 要明确记录或证明连续性后恢复。

[当前 draft.predict](../../../投机采样/scripts/p0_batched_verify.py) 真正输入是 current_state、previous_direction、sigma、next_sigma、timestep，加 checkpoint 的 normalization。B_on 必须保存**实际这些特征和同一 query state 的真实 target CFG 标签**；accepted draft 输出不能自己充当 teacher，另一条 target-only 轨迹的同一步标签也不能硬配。冻结 CFG、mask/projection、scheduler/cache 合同。保存已计算 teacher 标签不等于能够在缺少 KV/cache snapshot 时离线重放整个 target。

## 3. 第一版诊断合同（待实现/运行验证）

完整机器可读候选：[diagnostic-contract.json](../../../review/OMISSIONS.md)。

| 类别 | 必需内容 | 语义 / 验证要求 |
|---|---|---|
| 唯一关联 | campaign/worker/task/init/attempt/chunk/frame/action index，env-step、native simulator timestep、monotonic 时间 | 不伪造原生不存在的 server frame-id；可按已验证 trace context 关联 |
| 实际动作 | 本次传给 env.step 的 float32 七维命令副本 | 不再仅由历史文件重建；命令不等于位移 |
| 实际机器人状态 | 两个 gripper qpos；grip-site world pos 与 rotation matrix | shape/finite/frame 明确，单位验证；命令和实测分开 |
| 实际对象状态 | alphabet_soup_1 / tomato_sauce_1 的 world pos、native quat | 从实际 simulator id/name 绑定并 copy，非视觉颜色身份 |
| 接触 | 每物体左右 fingerpad 的接触及合取 | contact 不自动等于稳定抓持，力字段暂不要求 |
| 原生进度 | 每物体 In 布尔值、原生 done/reward | In 表示任务区域包含；逐步验证合取等于任务 success |
| RGB / 缺失 | 动作后 RGB 可分文件带 hash；缺字段 null+reason | terminal 也保存，不填 false/0，不混进 model payload |

N 个实际执行策略动作必须有 **N 个 post-action 记录 + 1 个稳定化后初始状态**；第一 chunk 跳过条件 frame、最后成功前缀仍按现有合同执行。原生 timestep 和 env-step 分开，五步稳定化不算 policy env steps。默认关闭；逐 worker 独立输出；先 copy 后缓冲落盘，记录采集异常和开销。诊断失败不能悄悄改变策略结果，也不能把缺失行当可靠标签。

“抬起”“稳定抓持”“掉落”“重抓”只能先做多信号派生候选，需验证相对位姿/双指接触的阈值与持续时间。不得把单一接触 bit、正命令、任意高度或 RGB 入篮当物理真值。

## 4. 分阶段最小对照：每一关验证后再继续

完整候选：[minimal-comparison.json](../../../review/OMISSIONS.md)。本步确认的是设计方向，不是以下执行许可。

### G1：先 client-only、默认关闭观察器与 CPU 合同测试

拟议源码范围仅 [client](../../../投机采样/scripts/p0_libero_multichunk_client.py) 的可选诊断 helper/开关和 env.step 后、done break 前的旁路记录；不改 WAM 模型、verifier、动作或现有默认行为。测试缺失值、数组copy、姿态约定、N+1记录、早成功terminal、worker/attempt join、开关 on/off 相同动作和模型请求、无额外step/infer/cache/RNG变更。

本轮 15 项 tests 仅验证窄源码语义，**不是此 observer 的实现测试**。具体修改与测试范围须得到明确确认，再实施。

### G2：真实 provider / 不扰动验证，然后极小完整诊断

先验证 runtime id/shape/finite、接触及目标值；用同 init、相同规定动作、已核对的初始状态比较 observer on/off，检查逐步状态/成功/步数一致。此项不假定完整 snapshot restore 已可用，也不把两个不同闭环当同状态。

建议诊断候选为 task0 的 init0（困难例）与 init1（参照），T target-only / S0 当前 K2，各两条完整 episode，**共四条，未授权或启动**。EGL＋独立并行 worker，两个 init 分配互斥；单 A100 条件串行、不争 GPU；完整现有 horizon、无人工截断，fresh provenance 与资源归属核查。

目的只是核实真实掉落/持稳/返工过程，不证明成功率非劣或速度收益。它是新增研究候选，**不自动解除旧 Debug-011 的执行门限/“最多极小配对”限制**；具体 runtime 预算和启动需单独确认。

### G3：先通过线上 teacher 数据合同，不立刻长训练

证明实际 predictor feature、query state、target label 的一一对应；覆盖全接受路径与 reject/fallback，标明 committed/off-route 节点；实际 previous_direction、sigma/timestep、normalization 与 mask 固定。双 head 是 proposal，不是现成概率。没有合格数据时不开始 B_on，也不从有偏 compact 日志伪造完整数据。

### G4：区分“训练更多”与“真实访问分布更好”

| 臂 | 更新内容 | 用途 |
|---|---|---|
| T | 冻结 target-only | 任务能力和成本参照 |
| S0 | 当前 draft 不更新 | 现有组合系统基线 |
| A_off | 从 S0 同预算继续离线 teacher 训练 | 控制额外优化次数 |
| B_on | 从 S0 同预算、系统实际访问状态 teacher 训练 | 测数据分布适配是否有用 |

S0 vs B_on 可以看新候选是否更好，但要归因“真实分布优于离线额外训练”，必须 A_off vs B_on 匹配初始化、结构、optimizer updates、样本预算；交互/teacher采集成本单列。target/verifier/threshold/global selector、plain K2/no新增bonus 与 scheduler/cache/stride 不变；不同时改 gate 和训练再归因。训练具体预算尚未确定。

### G5：真正任务反馈与 RL 留作下一项独立选择

- C1：用已验证关键持稳/返工事件加权 teacher 监督，可作低风险前置基线；它仍是监督、仍含 teacher loss，**不能称已实现 RL 或已脱离大模型输出训练**。
- C2：直接围绕组合系统 success、env steps、E2E 的任务反馈优化。当前 draft 无显式 log_prob；先定随机 proposal/参考分布或有依据的替代优化，再解决 accept/reject/fallback 信用分配，不能直接套 PPO。
- KL 只作辅助，不保证环境轨迹长度；固定同协方差 Gaussian mean KL 又是加权 L2。真实大小模型权重联训继续后置，不因本设计启动。

任务进度 shaping 先只列候选：两项 native goal count 必须运行验证；不代表持稳或连续距离。若用未折扣的 goal-count 差分，移出/重新入篮循环的差分相互抵消；有折扣时须另定一致公式。这个代数性质**不是成功/步数保证**。任何方案都不能把早失败的短轨迹当奖励改善，不能按反复抓放次数正奖励。

## 5. 评测合同：尚未指定数值，不抢先宣称非劣

主指标仍为 success/失败 → 配对 env steps 与 median/长尾 → 完整 E2E；accepted prefix、target rows/forwards、draft/verifier/fallback、cache/reset、L2 只解释结果。训练/采集成本和部署成本分开，诊断或teacher trace开启的 run 不充当 clean timing。

非劣成功容差、步数容差、tail gate、失败的公共 horizon 成本、训练/交互预算须预先确定，目前全部未定。所有 episode 都报告，失败不按“更少步”获益；同时单列双方成功配对但不做幸存者晋级。以 episode 为统计单位，不能把几千decision当独立成功样本。

init0–5 已用于诊断/查看结果，按 development 集处理。任何新 held-out 清单须先审查历史训练 manifest/选参与重用，区分“本轮更新未用”与“历史从未训练使用”；**不盲称 init6+ 必然未见**。四条诊断都成功也不能证明非劣；不能删除 init0、选最好 checkpoint 后在同几对上宣称泛化。

冻结参考身份是上一轮 source/provenance，而不是已锁定新 run 的权重清单：target robbyant/lingbot-va-posttrain-libero-long（registry 对应 libero_10 / LIBERO-Long），draft [当前 checkpoint](../../../review/OMISSIONS.md)，global threshold 0.007839680526450805，seed 20260922。draft SHA-256 本轮重新核对一致；target 完整 weight inventory 与 fresh source/runtime/device 校验仍是后续 preflight，不用旧 path 冒充全部权重哈希证明。

## 6. 产物、验证和明确下一步

- [source-feasibility.csv](../../../review/OMISSIONS.md)：7 个源码事实/provider 与未验证项。
- [source-audit.json](../../../review/OMISSIONS.md)：15 项窄 AST/CPU 检查和17个源输入hash；未导入完整 LIBERO/WAM、启动环境或模型。
- [frozen-reference.json](../../../review/OMISSIONS.md)：历史 target/draft/阈值/seed/config 身份；不是新执行许可。
- [diagnostic-contract.json](../../../review/OMISSIONS.md)、[minimal-comparison.json](../../../review/OMISSIONS.md)：设计候选，不是观测数据/训练结果。
- [validation.json](../../../review/OMISSIONS.md)：原始源码、旧证据/runtime、受保护状态及 HW 增量保留校验。

**下一步建议只到 G1：确认默认关闭 client 观察器的窄源码范围，然后实现并完成 CPU 合同测试。**完成后再单独确认 G2 的真实 provider/诊断采集；之后才 G3 数据合同与 G4 最小训练。每步输出继续提醒后面做什么。

持续入口：[RESEARCH.md](RESEARCH.md)；已发现的掉罐过程：[第 2c 报告](STEP2C-INIT0-VISUAL-REWORK-20261010.md)。
