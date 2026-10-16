# 最小冻结 target：失败时间线与成本诊断

记录时间：2026-10-11T07:50:14+08:00（Asia/Shanghai）。范围：**投机采样 / action-only，非 tile 算子**。

## 1. 本步完成情况

- 夜间实验已于 2026-10-11 02:04:39 完成（约 78.7 分钟），20/20 完整 policy episode；不是还停留在方案确认。此次仅分析其中 16 个评测 episode，未新启动 GPU、训练、真实环境 rollout 或服务。
- 四臂 T / S0 / A_off / B_on × task0、init2–5 的原始 episode、349 个 native chunk event、349 个 action tensor 全部唯一关联。按服务器 CPU 解码函数逐 chunk 逐位校验，重建 **5419 条实际执行控制命令**；条件帧和成功后未执行尾部不计入。
- 读取 5296 组动作后双视角 RGB，验证 10592 个数组。14 个成功 episode 的末 chunk 共缺 123 步 RGB；**两个 horizon 失败末 chunk 仍执行 cache，末尾 RGB 均存在**。不能套用历史“所有末 chunk 都没有 RGB”的假设。
- 失败各 796 个 policy env steps、simulator timestep 801；包含 5 次稳定化后在 chunk 边界检查 horizon800。native done 一直 false。这是现有完整执行合同，不是短截断或重复计数；自动报告的公共失败惩罚 795 与实际 796 分开保存。
- 不修改 target、draft checkpoint、verifier/threshold、bonus 或任何源码。旧 raw Knowledge、夜间 runtime 和 protected state/cycle/rules 均作 SHA256 不变检查。

## 2. 实验结果与训练范围

| 臂 | success | 失败惩罚后均值步数 | 总实际步数 | 总 chunk | 平均 E2E 秒 |
|---|---:|---:|---:|---:|---:|
| T | 4/4 | 280.00 | 1120 | 72 | 344.36 |
| S0 | 3/4 | 402.00 | 1609 | 103 | 453.38 |
| A_off | 4/4 | 273.25 | 1093 | 72 | 377.66 |
| B_on | 3/4 | 399.00 | 1597 | 102 | 474.00 |

A_off 相比 T：env steps −2.41%，E2E +9.67%；B_on 修复 S0 的 init4 失败，但新出现 init3 失败。**没有候选满足“任务可靠性 + 步数 + E2E”联合目标，不晋级部署。**4/4 不等于证明成功率非劣；init2–5 历史已查看，历史训练暴露未排除，仅 development pilot，不是新的 held-out 泛化结论。

训练两臂同 parent / 归一化 / 架构 / optimizer / loss，唯一数据预算各833，1000 updates × batch512，各512000 draws，固定最后一步。A_off 可用833条历史 target-only 样本，来自 task0/init0；B_on 可用2886条固定 S0 在 init0/1 真实访问的 predictor 节点（含未选中 branch），随机取相同833条。B_on 是**一轮真实访问分布上的 teacher 监督更新**，不是最终 candidate 自采样迭代、环境 reward 训练或 RL；target 没有训练。训练 loss 也不是共同 held-out 任务指标，两臂不同数据的 loss 不直接比较，更不能用下降证明更少步。

## 3. 排序假设与检查结果

| 假设 | 验证与反证 | 当前判断 |
|---|---|---|
| H1 关联/stride/终止合同造成假失败或假步数 | 16 episode、349 native event 与解码/动作前缀/累计 t 对账；两个失败确为完整 horizon、done false | 已审查的记录/计数错配未发现；不排除所有模型/cache 数值问题 |
| H2 少走步自然等于少算 chunk | A_off 四个配对分别少1、13、9、4步，但 chunk 均与 T 完全相同 | 该推论被本轮反例否定；必须另看 chunk 与实际工作预算 |
| H3 reset 或并发需求差异解释慢/快 | reset/policy/cache/残差可闭合；A_off 主要多在 policy，S0 init5 整局无 peer policy 需求重叠 | 请求成本与并发混杂已定位；真实 target forward/接受树工作原因未测 |
| H4 抓取/持稳/返工决定长尾 | B_on init3 可见罐掉落；S0 init4 对象未成功提离、手部离开后罐留原位，后续均反复尝试 | 物理行为现象有图像证据；接触、关节、物体 pose、In 谓词未记录，尚无根因归属 |
| H5 真实访问分布训练应稳定改善任务 | B_on init4成功，但init3失败；仅init0/1采集，不能验证困难阶段覆盖 | 此最小配置未显示稳定优势；不能据此否定所有在线蒸馏，也不能宣称分布已覆盖 |

## 4. init3：B_on 第二次搬运中可见掉罐

| 臂 | 完成 | 步数 | 第二段闭合命令区间 | 本步观察 |
|---|---|---:|---|---|
| T | success | 300 | 232–292 | 相应窗口中对象提起并随手部移向篮子，最后原生 success true |
| S0 | success | 290 | 227–282 | 相应窗口继续持物搬运，最后原生 success true |
| A_off | success | 287 | 225–280 | 相应窗口继续持物搬运，最后原生 success true |
| B_on | failure | 796 | 217–279 | 约235–238步绿色/灰色圆罐离开夹爪并落向篮子外桌面；268步已侧躺，后续尝试未完成任务 |

B_on 224–240逐步窗口中 command6 始终为正（约+1），没有在掉落前切成打开命令；其后直到279仍为正。因此“掉罐就是因为提前发了松手命令”**不能解释这段记录**。217开始闭合比T的232早15步，是不同轨迹的描述，不是同状态反事实或因果证据。闭合命令不证明夹爪实际闭合程度、双指接触或持稳。

B_on 全局夹爪命令符号翻转14次，T/S0/A_off均4次；失败长尾中多次靠近/开合但最终 done false。这只能说明反复尝试，与不同轨迹长度有关；不将“次数多”当作根因或可靠 reward。

细图：[B_on init3 224–240](../../review/OMISSIONS.md)；[T init3 212–252](../../review/OMISSIONS.md)；[全程 B_on init3](../../review/OMISSIONS.md)。图像按原像素或明确的最近邻放大，未合成/补帧。按外观标记对象，未绑定 simulator object ID；成功末 chunk RGB 缺失，不补画最终入篮。

## 5. init4：S0 更像未建立稳定抓取/提起，而非先假定搬运掉落

| 臂 | 完成 | 步数 | 第二段闭合命令区间 |
|---|---|---:|---|
| T | success | 298 | 235–291 |
| S0 | failure | 796 | 243–336 |
| A_off | success | 289 | 224–280 |
| B_on | success | 294 | 235–288 |

S0 220–260窗口里，绿色/灰色罐位于手下；随后260–284手部离开，而罐仍在桌面原位置附近。T 同期已持物向篮子搬运，A_off/B_on也完成任务。S0 长尾中继续出现开合/往返，总符号翻转16次，其他三臂4次。**图像支持“第二次抓取/提起未完成，随后返工”的候选解释，但不能辨认究竟没抓实、短暂接触后滑脱、运动控制误差还是其他接触机制。**不能将它与B_on init3直接写成同一种已证明的掉落原因。

S0 第二段正命令到336才结束，比T的291晚45步；这与未完成搬运一致，但不能用绝对 env-step 对齐推导闭合时机的因果作用。原生 success false 可以确认任务未完成，缺少分别的两个 In 谓词，不能确定全部后续阶段的 goal 状态。

细图：[S0 init4 238–254](../../review/OMISSIONS.md)；[S0 220–260](../../review/OMISSIONS.md)；[T 220–260](../../review/OMISSIONS.md)；[S0 260–284](../../review/OMISSIONS.md)；[全程 S0 init4](../../review/OMISSIONS.md)。这些是所检查窗口的视觉定位，不声称找到所有数值分歧的第一次或物理根因。

## 6. A_off 为什么步数减少却更慢

四对 T→A_off 步数：254→253、300→287、298→289、268→264；**chunk 都是17、19、19、17**。每次首chunk容量12、其后16动作。少27步主要缩短各成功末chunk的执行前缀，仍需算该chunk的整次 policy 请求；未节省任何一次 policy/cache 请求。

| 计时项 | T 均值秒 | A_off 均值秒 | 差值秒 |
|---|---:|---:|---:|
| reset request | 123.39 | 119.92 | −3.47 |
| policy requests | 192.25 | 227.87 | +35.62 |
| cache requests | 9.28 | 11.19 | +1.91 |
| 剩余客户端时间 | 19.44 | 18.68 | −0.75 |
| 总 E2E | 344.36 | 377.66 | +33.30 |

累计 policy/累计chunk：T **10.681s**，A_off **12.660s**，+18.53%。两臂均72chunk，缓存均68次；A_off增加时间主要在policy请求，不是reset更慢。client websocket monotonic请求计时包括服务器计算/排队/通信，并非纯GPU kernel time；残差含环境执行/日志/其他客户端工作，不叫作纯环境时间。

T/A_off四个episode请求阶段与peer策略需求包络重叠比例约0.925–1.000，无chunk是在peer最后episode结束后启动。比含失败臂更接近相似并发，但不同轮次/状态/接受树/CPU-GPU竞争仍不相同，不能证明多1.98秒全来自draft算子或某一verifier分支。评测未记录accepted depth、精确target forwards/rows及子阶段profile；普通K2与bonus不可混算，本次bonus未开启。

## 7. 必须修正的 timing 解读：两个独立 worker 不等于恒定双并发

- S0 worker0在01:27:40.994结束init3；worker1于01:27:41.514开始init5。**S0 init5全部17次policy在peer最后episode完成后开始**，累计policy/chunk仅5.784s，E2E239.26s；T init5为10.567s、323.61s。不能把这84.34秒差值当作相同并发下的S0模型收益。
- B_on init3后29/50chunk在peer最后episode完成后开始。server `Infer One Chunk`到client执行完该chunk的时距，在这些尾部平均约9.266s；完全位于peer policy需求包络内的15chunk约16.411s。该时距含环境动作执行，不是逐chunk纯推理计时；前后物理/接受状态也不同，不能做因果速度比。
- S0 init4虽没有chunk在peer最终结束后开始，但peer在切换episode时做reset；其13个chunk完全不在peer policy需求包络中。双模型服务器仍在、peer reset也可能消耗CPU/GPU，所以“无peer policy”不能当作GPU完全空闲。
- 需求包络定义为每个episode首policy日志至client完成，含内部cache、环境执行和间隙，是**并发需求代理**，不是实际请求逐次重叠或GPU利用率。输出窗口CSV、worker时间清单供复核，不以这个代理校正/替换原始E2E。
- 四算法臂仍串行、每臂2独立EGL worker，没有混条件共跑。完整episode实测时间仍是该调度下真实成本；但**聚合E2E不是严格同负载的算法因果比较**。尤其失败会改变后续worker负载，这是比较需要控制的问题。
- 本次关闭native observer/teacher/full trace/profile作评测，但所有臂都保留chunk事件写日志。称为“诊断旁路关闭的计时”，不是完全零日志无扰动性能基准；共同事件日志开销在残差内，未单独校准。

## 8. 决策及下一步（未执行）

1. **不继续盲目加训练，不上RL/KL或真实大小模型联训，不改验收阈值，不部署晋级。**此次只能说A_off步数方向较好但没转化成chunk/E2E收益；B_on本配置没有稳定提升。不能说“真实LIBERO训练无效”，也不能说“L2下降已解决任务”。
2. **下一步优先原生诊断，不再扩大训练：建议单独授权6个完整episode，T/S0/B_on × init3/4。**每臂2独立EGL worker（各持一个不重叠init）、算法臂串行，沿用相同checkpoint/seed/协议/global阈值/horizon，无bonus和额外target forward。记录两物体pose、grip-site/qpos、双指contact、各自In篮谓词、done及terminal RGB，核对命令/图像；特权字段仅用于诊断，不入部署输入。小样本仍开发集，不能证明泛化。
3. 原生诊断必须先确认旁路开关合同/实际provider/RNG检查仍通过；此前只验证16步规定动作的读取不扰动，**完整真实WAM on/off数值等价未证明**。失败先停，不扩大预算、不强行训练。对于接触/持稳，先验证字段和阶段定义，不能拿未经校验的标签直接做reward。
4. 成本线随后单列：先设计满足独立并行worker约束的T/A_off同策略请求负载比较，预先匹配phase/并发需求，条件交错，分开reset与稳态；默认关闭的accepted-prefix、target forwards/rows和draft/verifier/scheduler子阶段计数只作诊断，并先验证不改变推理。具体屏障/调度方法还未实现，不能谎称已经拥有公平逐chunk成本比较。
5. 只有原生字段确认敏感阶段/返工机制，再定阶段或关键动作加权、混合teacher+任务反馈、困难状态再采样的最小对照；仍冻结target，task success→步数/长尾→真实成本为主。RL概率/信用分配/KL定义与正式非劣容差仍待另一步设计。

**本步已完成数据对齐、所选窗口视觉定位和成本账本；尚未完成物理/模型因果归属及同负载成本验证。后续6episode诊断超出已完成夜间20episode授权，未自动启动。**

## 9. 证据与可复核性

- 夜间报告：[HW结果入口](../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)；原始运行目录：[runtime](../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242)。原始数据不回写。
- 本步 [summary](../../review/OMISSIONS.md)、[episodes](../../review/OMISSIONS.md)、[chunks](../../review/OMISSIONS.md)、[actions](../../review/OMISSIONS.md)、[observations](../../review/OMISSIONS.md)、[命令符号区间](../../review/OMISSIONS.md)。命令为保存tensor按固定解码合同重建，不是额外实测夹爪/机器人状态。
- [成本汇总](../../review/OMISSIONS.md)、[配对差分](../../review/OMISSIONS.md)、[并发需求时间清单](../../review/OMISSIONS.md)、[图像窗口及源hash](../../review/OMISSIONS.md)、[输入hash](../../review/OMISSIONS.md)、[验证](../../review/OMISSIONS.md)。
- CPU脚本/可复核校验一并归档；torch weights_only=True，窄NumPy safe_globals，CUDA_VISIBLE_DEVICES为空，单CPU线程。未导入WAM模型或启动模拟器；没有新GPU进程。单agent不同脚本对账，不称独立第三方复现。
- source/旧raw/protected/夜间runtime共2479个文件hash保持不变（目标大权重只继承原run路径/size/mtime清单，非全权重content-hash冻结证明）。保存新报告前统一secret-safe redaction；tile旧里程碑状态保持原值。
