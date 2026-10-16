# 下一阶段扩大实验的准备条件

2026-10-15，Asia/Shanghai。本文件是实验执行说明，不是论文初稿。夜间最终选择以[统一结果](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)及其实际闭合证据为准；本文不替代正式运行计划或授权。

## 结构规格与部署资格分别记录

已可精确复现的候选结构是 `draft_architecture_v1/models.py` 的 `transformer64`：16个动作位置，每位置7维state、7维previous direction及3个时间标量，共17维；Linear17→64，学习位置编码，一层pre-norm Transformer、4 heads、FFN256、GELU、dropout0，再Linear64→14。输出为center±delta两个对称proposal，解码为previous direction加反归一化residual。3个标量是sigma、next_sigma、timestep/1000；这里sigma是flow scheduler坐标，不是直接套用DDPM噪声标准差。完整模型53,070参数；对照MLP96为52,928参数，现用A_off为25,928参数。

结构参数已由CPU实际实例化复核，并保存[可读取结构合同](../../../review/OMISSIONS.md)；没有修改部署preset或授权新权重。

这是下一批研究数据和匹配训练的明确结构方向。是否换成部署权重，仍取决于至少2/3 seed通过原离线全部门、所选checkpoint自身通过、原生完整chunk验收，以及新完整任务对照。现有CPU-exact适配器用于诊断采集，包含CPU/GPU拷贝；尚无新TF64的GPU延迟或加速证据。冻结OLD加零头TF的残差版总runtime参数78,998，不应标成53,070。

## 先扩大训练数据，再扩大结论

下一批数据应扩到LIBERO-Long全部10任务，并预先登记TRAIN、开发DEV和最终评测的任务/init/seed使用情况。保留所有完整训练episode，包括失败与超时；标签来自候选实际访问状态上的同查询teacher，不只重放历史轨迹，也不按接受率或成功筛样本。初始化编号不能仅因未出现在某一批72份plan中就宣布全历史未使用。

已完成的[只读溯源](../../../review/OMISSIONS.md)证明标准500个task/init组合均有历史completed记录，但不证明全部用于训练或调参。A_off有父task0–7训练和task8选模继承；TF历史池已把原task9/init0测试数据用于训练。可以继续冻结并报告标准全50初态benchmark结果，需公开用途和历史接触；更强的独立测试声明不能仅靠另取init20–49。新协议应先核对各集合参与训练／选型的实际用途，以及固定测试随机种子和停止规则，当前未认证新TEST。

维持已审查的匹配训练比较：TF当前辅助、TF新查询辅助、MLP96新查询辅助；current512暴露和KD相同，另128条DATA，loss为current_C+0.25×auxiliary_DATA，一次640条forward，三个seed，各1000updates，lr1e-4、weight_decay0.01。归一化只从TRAIN得到，DEV不更新参数或normalization；最终评测不能参与结构、学习率或阈值选择。扩大数据的有效性需重新测试，不预言全量训练一定修复保持问题。

本轮45次执行对应14个非重复配置、42个配置×seed单元，只有三个seed策略。所有六门、控制复用和实际耗时见[选型全表](../../../review/OMISSIONS.md)。

本轮λ4/16、冻结OLD残差、低学习率均没有产生稳定替换配置。下一轮仍先做上述小规模匹配对照，避免直接把全部训练预算投入一个未通过门的checkpoint。若三seed仍失败，需分析逐task/denoising-step的旧通过丢失与候选访问状态覆盖，再设计新的训练目标；本轮没有实测任务奖励PPO训练收益。

只读检查7767条旧TRAIN／DEV／新TRAIN后，没有精确重复的227维输入，也未发现相同输入异teacher的冲突。不能拿输入冲突或两head覆盖不可能解释当前失败；更大数据或新的条件特征是否有效都需要额外实测。

## 冻结算法后再做独立质量和计时

完整chunk的50个scheduler更新必须保留；从环境初始化到成功的动作数无需人为做成相等。评测应同时报告配对成功损失、成功率、失败/超时、全部episode的失败惩罚步数，以及共同成功pair的长度比及尾部。只报告成功轨迹的平均长度会遗漏失败变化。当前开发门是成功损失0、总惩罚长度比≤1.05、最差pair≤1.10，属于本项目约定，不是文献给出的普适阈值。

接受规则的优先研究候选是原步长加权L2与连续前缀，保留现用OLD；greedy可以进入下一轮冻结比较。相同目标的DP已与现穷举完全等价。严格逐位后继门没有接受近似边，并不能消除B14 verifier相对CFG2教师的整策略数值差异。PF2表示2个投机chunk后一个原大模型chunk；bonus是单chunk内消费已验证后继的teacher更新，包含在原50次积分更新以内，不额外增加第51个积分步。原T的51次action调用还包含不推进积分的terminal调用。PF与bonus不能混称。

同原τ的PF2／PF3／PF4已各在两个复用DEV pair完整成功，相对T合计长度+1.277%／+0.912%／+1.095%；target调用均多于连续OLD（+13.38%／+7.52%／+4.64%）。当前保留OLD＋bonus；无刷新改善总体成功率或延迟的证据。

只有在空闲GPU及同一模型/状态/CPU配额的正式配对计时中，才比较新draft、greedy或PF的速度。已有1.602×/1.544×/1.500×仅对应当前preset的短/中/长KV完整policy请求，排除reset和环境步骤。质量实验的target调用减少、双worker周转和CUDA数值前检不能代替新的速度结果。

## 有资源条件的时间预算

已完成24个质量episode，含25次冷模型加载、reset、双EGL、固定数值前检和清理，阶段实耗6522.859942 GPU秒（单张分配GPU、双EGL阶段elapsed，包含六臂混合周转）。因此按这批摊销线性外推：50个episode约3.77小时，100个约7.55小时；Long10×50init×教师/候选两臂共1000个约75.50小时、3.15天。这是两任务观测的外推，另外8任务和800步失败的耗时尚未测，不是完成承诺。持久复用模型可能减少加载开销，但本轮未实测该改动。实际九组CPU训练加完整复算约9–14分钟，正式新阶段仍需prepare、独审及原生适配验证。

四个开发pair不能证明总体非劣。例：固定评测总体、独立配对、零成功损失事件时，单侧95%二项上界为1−0.05^(1/n)；n=4时为52.7%，n=149时约1.99%。这只是成功损失事件概率的保守统计例子，不能把开发集、多候选选择或每任务结论混进去，也没有替用户新定义2%准入门。

若需要AutoDL，建议选一张H800 80GB并配足CPU，按当前并行方案至少16个逻辑CPU；这是减少环境改动并保留双EGL／CPU训练／独审配额的资源建议，不是已测的H800吞吐。平台说明GPU独占且CPU／内存随租用GPU配置分配，实际选择应看实例CPU与内存配额。[AutoDL官方选型说明](https://www.autodl.com/docs/gpu/)。当前reset占请求wall约54–61%，不能仅用GPU峰值算力预测周转；H800尚未接入。迁移时固定PyTorch2.9＋cu126／Python3.12等依赖，复查原native扩展的目标架构和新机数值；nvidia-smi显示的是驱动支持版本，不能当实例实际工具链。[AutoDL官方CUDA说明](https://www.autodl.com/docs/cuda/)。未发起租卡、远程安装或SSH连接。

先完成Long10，再迁移到有官方LingBot-VA权重的RoboTwin局部任务，然后扩大两模拟器的固定评测。RoboTwin的3视角、16动作通道和normalization与Long不同，需重新训练draft和校准，不能直接复制Long阈值。RoboDojo匹配的公开VA权重尚未核实，暂不放进关键交付路径。Fast-WAM作为第二模型的价值高于立即增加第三模拟器；它是单独的适配和实测工作，当前尚未复现。

算子侧保持暂停。可同步开始中文方法段落和已实测结果整理；用户重新安排verifier优化后，再依原tile研究做独立profile/原型/数值/速度，不能把本轮撤回的融合原型当作已有性能贡献。CVPR2027内部完整稿安排11月1日18:00给老师，官方全文截止11月16日AoE；若10月21日前独立质量没有达标，应缩减扩展范围并保留失败事实，不把未测收益写入稿件。

证据：[CPU训练、闭环与官方资料汇总](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[时间及统计复算](../../../review/OMISSIONS.md)、[模型/基准官方核查](../../../review/OMISSIONS.md)、[原结构代码](../../../投机采样/experiments/draft_architecture_v1/models.py)。
