# 当前O3旁路teacher与下一轮数据：未执行设计

当前只实现TF_CONT，以下新采集与TF_ON/MLP_ON入口仍HOLD。原源码不改。本文件没有把离线局部分歧当作闭环失败原因。

## 可达定义

保留每树全部3个同query teacher节点，但显式分三个标志：

- root始终`state_reached=true`。被选中的parent只有`accepted_depth>=1`才`state_reached=true`；depth1也到达parent，因此应能使用该parent的target监督，即使其第二条draft边未接受。
- `selected_branch`是路线选择标记，不能证明到达：depth0时选中的parent仍是反事实。
- `draft_edge_committed`沿用旧语义：root节点对应的出边depth>=1才提交，parent对应的第二出边只有选中且depth2才提交。它不能替代`state_reached`。

拟新训练只取FIT中的root+真实到达parent；非选中parent、depth0的所选parent仍归档为counterfactual，默认不混入主训练。不得在CAL看效果后改变这个规则。此方案尚未证明比全3节点训练更好。

## 保持当前O3的最小接线

旧`draft_onpolicy_train_v1/collector.py`复用原`p0_state_teacher.teacher_rows`，确实从已算verifier输出取target，extra_teacher_forwards=0。但直接设`server._p0_state_teacher`会触发`prefix_acceptance_v1/measured_online.py`的`trace_or_teacher`分支，action从O3 workspace转到scalar base；所以不能直接启用它并声称当前O3未变。

新collector必须保持`_p0_state_teacher=None`及`_p0_trace=None`，在独立server实例做只读旁路：

1. 包装实例`_target_action_candidate_directions`，调用原函数且原结果原样返回；在同一调用从实际`candidate_states[:3]`提取compact query拷贝、frame/timesteps和当前request身份。不得再调用target。不要通过包装score函数破坏`scalar.api`的原数学AST检查。
2. 包装`selection.ObservedAPI`实例的`execute_k2_b2_route`方法。`scores.target_directions`已由原score保留7行compact target的clone（p0_batched_verify.py）；不需重新score或target。调用原execute并得真实accepted_depth，之后在下棵树/下次workspace生成前立即clone root、root_previous、两parent、两first_direction以及前三target。对state与第1步记录的真实candidate query逐位核对（有限BF16可转uint8），不能只因同node编号就认为同query。
3. step_index从实际action调用帧的locals只读取得，或由原actual timestep表唯一匹配并与context+tree序号联合核验；不能用计数/推算接受深度替代真实step。优先沿已审核S04方式用调用栈真实locals，但该方法必须事前AST断言locals名及唯一原调用点。
4. 复制完成后转CPU FP32紧凑列，每个chunk批量atomic保存，记录源dtype、完整sigma_path/target_timesteps、parent/child/depth、root/parent reached标记、proposal输出、原FP32六score/τ、generator child23字节、target权重登记、episode/task/init/chunk/frame/tree_step/node。不要持有O3 workspace别名到下次build。保存失败使采集资格FAILED，不修改当次原策略返回；完成全部轨迹后检查每chunk保存树数=COUNTER tree_count及逐局守恒。
5. 正式新30局前，应以已有完整recorded episode做teacher OFF/ON完整重放验收，逐位核actions/原route/target计数/有效KV/RNG，额外teacher forward必须0；这属于完整真实验收，不是smoke。若原latents未保留，不宣称与旧视频latent逐位相等。

原API数学和生产source不替换；只实例包装及保存。该旁路尚未实现、未通过GPU数值验收，不能把设计当collector已可运行。

## 四局已有视觉的用途与边界

本机OBS-DIAGNOSTIC-PRESERVATION/ALL-CONTENT-AUDIT证明四局102个obs_data文件仅含两路uint8相机，7个tar完整保全。对应T task1/init29、TF同局，T task6/init25、TF同局；另完整small档有COUNTER/actions/commands/初始观测与RNG，但**无latents/KV/原velocity**。

旧Q02 worker硬编码fresh task7/init24的前10chunk、必须先逐位复现原actions+latents+routes，再进行7seq对照。不能原样指向这四局：输入manifest/完整episode长度/四局真实RNG来源要替换，且旧latent等价门缺证。可用新namespace对两条TF完整轨迹共71个chunk按记录观测重放（T两局31chunk另作原模型参照），先严格actions/routes匹配；如不匹配即不取得“旧真实TF同query”资格。离线重放即使匹配，也不是新的闭环机器人episode，不能解释历史失败原因。

保存teacher自身额外target前向为0；**重放完整71/102个chunk仍会重新执行真实模型，不能把总重放成本写0**。已有init25–29只用于DEV诊断/接口验证，不能混入下面新FIT以制造独立对照。

## 下一次固定30局（未采集）

生成策略固定当前TF child23，原K2/B2 bonus、O3、τ、调度、7通道不变。task0–9×init30/31为20 FIT，task0–9×init32为10 CAL，全部成功或horizon终态都保留，不按结果筛。每局真实target初始化RNG/两相机/初终物理状态/完整acceptance+命令+保存覆盖。CAL只评估，不optimizer/归一化拟合/训练抽样；未来TEST40–49不读。

每树root+两parent的teacher保存不加target forward；仅CPU拷贝/存储/校验开销。紧凑state/previous/target FP32为每行1344B，再加时间与整数身份约几十B。每chunk一个列式shard比每树一个torch.save更省空间/文件数，不保存全KV或latent。按当前新50 TF实测1049chunk/18896树，30局类比约629chunks、11338树、34013全节点、21149可达节点，纯三向量约43.60MiB。此为按当前分布的容量估算，不是新数据结果。

保守每局最多50chunk、每chunk不超过50棵树：30局≤1500chunk/75000树/225000全节点；纯三向量288.39MiB。建议teacher+索引硬帽512MiB，整采集输出（含已有policy小产物与必要初态）独立≤2GiB；正常不留所有obs/视频/大trace，若要可重放全视觉需另登记约2.5GiB级预算，不能隐入teacher帽。达到储存/forward/墙钟帽即PARTIAL，不补合成行、不称完整30。单模型+单EGL持续运行，不能复制旧双server33GiB资源编排。

## 三组500更新对照

- TF_CONT：当前TF child23的2500起点；原历史target256+原samequery256再500；本目录已实现，3000新schema。
- TF_ON：完全相同TF child23起点、AdamW重置、旧normalizer、原loss/lr/BF16评估，历史target256+新FIT可达节点256，500更新。唯一主变量是samequery来源。不能把TF_CONT改成512 target-only，否则同时改变原混合配方。
- MLP_ON：原MLP child**23**（非selected22）2500起点，完全相同target/newFIT采样draw与500×512预算；MLP96 52928参数、TF64 53070，差142（约0.268%）。两结构初始权重无法逐位相同，只能声明同seed/训练历史/预算。生成数据来自固定TF23，所以它是“MLP使用TF诱导分布的匹配数据对照”，不能称MLP自身on-policy。

三组固定主seed23，不按CAL改seed；若以后扩seed22/24，三组对每seed共享draw/同结构同初始checkpoint，另事前登记，不能挑最好seed。TF_ON/MLP_ON loader必须新schema绑定3000/真实父2500/原normalizer与新FIT/CAL证书，不修改旧two-stage2500断言。CAL评估同时保留根/真实到达parent、σ段和原CAL20237遗忘诊断；局部通过不授予机器人成功率资格。最终要在未训练的DEV33/34做原target/当前TF/cont/newTF/匹配MLP完整闭环对照，固定horizon与真实初态配对。
