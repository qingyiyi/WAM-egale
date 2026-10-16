# Debug-017：稳定600步轨迹的第二次抓取返工与成本对照设计

日期：2026-10-11T13:04:23.468518+08:00（Asia/Shanghai）。范围：投机采样/action-only，**非tile**。用户同意继续CPU轨迹分析及同负载T/A_off设计；本轮未新启动GPU、真实环境、服务、训练或修改公共模型/控制源码。

## 1. 本步完成情况与核心结果

- **阶段定位完成**：已保存S0 init4 600步轨迹的主要延迟，集中在第二罐抓取建立之前，不是第一罐入篮后掉出或第二罐成功提起后反复掉落。
- **成本设计完成，执行未完成**：已有profile/counter可用性与15项静态/纯CPU调度模型合同核查通过；真实同步包装/逐请求边界记录尚未实现，新的同负载WAM或E2E实验均未启动。
- 本轮只读取2条原生轨迹（init3/4）；先前OFF-A/OFF-B/ON是同两起点重复/开关验证，不能当6个独立泛化样本。CPU对账895原生行、893执行动作、24组原始双RGB及729个去重文件hash通过。
- **未证明**第二次抓取误差由draft、L2、验收策略、某轴或某cache造成；也未证明相同init的target-only会更短。新旧RNG/PYTHONHASHSEED/共同指纹合同不同，不能把当前600步与旧310步当单因素因果反事实。

## 2. 长轨迹到底多在哪里

以原生首次soup In、首次tomato双指contact、终点success作为描述性标记。In是区域包含，不等于释放落稳；双指contact是采样接触，不等于稳定抓取。本轮soup首次In之后不再变假。

| 轨迹阶段 | init3（293步） | init4（600步） | init4−init3 |
|---|---:|---:|---:|
| 开始→第一罐首次In | 168 | 167 | −1 |
| 第一罐首次In→第二罐首次双指contact | 69 | 359 | +290 |
| 第二罐首次双指contact→success | 56 | 74 | +18 |
| 总步数 | 293 | 600 | +307 |

+290约占两条轨迹描述性总差的94.46%，将后续诊断聚焦于第二次接近/建立抓取，而非全局平均L2。**两个init的物体起点不同**，这个比例不是K2相对target多走步的因果分解。阶段窗口还包含第一罐从In到释放的交接，不把每一步都叫失败抓取。

第二罐首次双指contact为init3 step237、init4 step526；初始高度+1cm标记为241/531，+5cm为246/536。1/5cm是描述性阈值，不是LIBERO原生奖励。init4最终soup/tomato In均真，目标首真step600；没有修改成功定义。

跨阶段chunk会重复相交，不能把intersecting_chunks求和当总chunks，更不能将半个chunk的已执行步数直接折算半次policy成本。[分解明细](../../review/OMISSIONS.md)。

## 3. 四次没有建立双指接触的闭合区间

cmd6>0仅表示闭合指令，不表示实测夹持力或抓住了对象。表内any包含单指与双指接触；数值取每个完整闭合区间，不误称全部发生在step526之前。

| init4闭合区间 | 长度 | 采样双指contact步 | 采样任一指contact步 | 最高抬升mm | 最小EEF-site/物体中心距离mm |
|---|---:|---:|---:|---:|---:|
| 242–299 | 58 | 0 | 0 | 4.05 | 43.99 |
| 342–367 | 26 | 0 | 0 | 5.47 | 50.04 |
| 392–425 | 34 | 0 | 2 | 6.55 | 52.04 |
| 478–489 | 12 | 0 | 0 | 0.65 | 50.08 |
| 523–591 | 69 | 66 | 66 | 227.79 | 8.34 |

- 前四段242–299、342–367、392–425、478–489，分别58/26/34/12步，**双指contact均0**；第三段有2个单指样本，其余三段闭合期间any为0。闭合前还存在left contact232–241、389–393，故不能写“此前从未碰到第二罐”。
- 到step525之前，第二罐相对初始高度最高仅6.55mm，中心位移最大15.56mm；图像中罐多次留在桌面、夹爪离开又靠近。这支持未建立有效提起/反复尝试的定位，不支持“已经稳定搬运后掉落”作为这段长尾的解释。
- 第五段523–591，526开始双指contact，随后明显抬升；双指区间526–597，left可到598。最终600满足原生目标。init3第二罐仅233–293一段闭合，237起双指contact并搬运至成功。
- 最小center-distance时，前四段世界坐标EEF-site比罐中心高约41.0/48.5/47.9/49.5mm；第五段约7.1mm，init3成功段约8.3mm。前几段关节slide差最低约1.0/1.5/1.1/22.7mm，而成功段约63.8mm。这是**抓取接近位姿/夹爪空合的线索**，不是指尖净空、实际力、判定阈值或已证因果；eef-site/center不是相同物理表面。
- 状态每env.step后采样，不能排除单个env.step内部的短接触。此次没有指尖pose/力、完整控制器目标或接触力，不能把“未采到双指”升级为所有微时间点从未接触。

原始RGB与native字段绑定同step，24组双RGB逐数组hash核验；图板仅显示原数组/最近邻显示，不合成补帧。success前缺的末chunk图像没有填补；只使用已存动作后RGB与293/600实际terminal NPZ。

![原生动作与接触阶段](/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/init4-grasp-rework-and-cost-design-20261011/stage_timeline.png)

[init4原始图像](../../review/OMISSIONS.md)、[init3参考图像](../../review/OMISSIONS.md)、[原图来源与hash](../../review/OMISSIONS.md)、[闭合区间几何](../../review/OMISSIONS.md)。

## 4. 排序假设与结论

| 假设 | 核查/反证 | 当前结论 |
|---|---|---|
| H1 第一罐未放好/反复掉出导致长尾 | soup In167起一直真，第一阶段只比init3短1步 | 这条轨迹的主要延迟不在第一罐原生目标反退；不证明已释放落稳 |
| H2 第二罐未建立抓取/空合/返工 | 四闭合段无采样双指、最高仅毫米抬升、图像罐留桌面；526起才双指且明显提起 | 当前最有证据的阶段定位；微观物理与模型原因待证 |
| H3 搬运中提前发打开导致反复掉罐 | 前四段指令仍为正且没有明显搬运，成功段后才完成任务 | 对这段长尾不支持早开导致已持物掉落解释；不泛化到旧B_on失败 |
| H4 闭合前位姿/深度误差是draft特有 | 失败段EEF-site明显更高、夹爪关节差变小；但只有S0两不同init | 候选假设，须同init T/S0反事实与同candidate teacher数据；不能直接归因z或L2 |
| H5 观察器/计数错配制造长轨迹 | 前步OFF/OFF与ON/OFF全已测轨迹/保存张量一致；本轮命令/连续行对账通过 | 当前测试范围不支持该解释；未证明所有hidden状态 |

## 5. 成本账本的新增实现边界

源码核对确认：已有WAM_P0_PHASE_PROFILE在边界CUDA同步，counter仅在active profile有效；不是独立轻量计数器。T普通路径从phase.calls核对target调用，不能读空counter得出0target。profile只覆盖policy，reset/cache内部细项缺失保持缺失。

当前普通K2仍每tree验证7逻辑行，CFG因video guidance触发而扩为14实际候选rows（action guidance=1不关闭共享use_cfg）；leaf消费/3行优化不在本任务。实际calls/rows是工作账本，不是FLOPs或按batch比例预测时长。

新的请求cohort/双worker到达屏障/每请求server边界尚未实现。CPU只验证静态路径和19/38、19/19、51/19的队列模型不截断/保留single-demand尾部；没有给现有结果贴上“严格同负载”标签。

[成本对照草案](../research/speculative-sampling/COST-MATCHED-DESIGN-20261011.md)具体区分：同init物理对照、T/A_off双需求诊断、另一次profile-off自然E2E。profile-on、屏障等待、单worker尾部与真正部署E2E分表，不直接比较不同phase或不匹配样本。

## 6. 对后续训练思路的影响

- 先以第二罐**闭合前接近位姿与建立抓取**作为困难阶段候选，不继续用平均L2/接受率替代任务进度；L2保留局部辅助。
- 未来同candidate冻结target监督，需要实际去噪predictor features/x_i/previous与同query的target标签；本次893部署动作没有这些teacher features，**不能直接当draft训练样本**。
- privileged pose/contact可用于诊断或辅助监督，但不能偷偷输入部署policy。接触是测量事件，非稳定抓持reward；不能仅把“有contact”加奖励便说能缩短轨迹。
- 只有同init对照确认额外损失，再定任务敏感维度/阶段加权、困难状态再采样或混合teacher+任务反馈。target继续冻结，小模型任务成功率/步数/chunk/E2E主验收；RL的密度、KL定义与信用分配仍待后续，KL不保证trace长度。

## 7. 下一步与授权边界

**推荐下一关最多4完整episode：T/S0 × init3/4，在同一新seed/首局/观察器合同下作原生配对。** 先确认任务包装范围、预算、当前GPU资源与EGL/端口；profile/teacher/full trace/训练关闭。fresh S0需复现已保存293/600与关键事件，否则先定位分叉停止，不自动加局。

这一补充先判断“第二次抓取返工是否相对T新增”；成本T/A_off方案已完成设计，但同步包装和候选8局诊断预算**另确认、另关卡**，不与4局绑定自动执行。当前没有启动任一新GPU任务，也没有停止等待中的tile任务。

若T也长，先查共同起点/控制/任务；若T短而S0长，重点查同candidate关键抓取动作再定冻结target反馈训练；若S0重复不稳，回到复现。无自动RL、KL训练、大小模型联训、bonus/验收阈值或公共源码修改。

## 8. 验证与记录保护

[CPU复核](../../review/OMISSIONS.md)、[统计](../../review/OMISSIONS.md)、[逐步原生字段](../../review/OMISSIONS.md)、[原生事件区间](../../review/OMISSIONS.md)、[阶段统计](../../review/OMISSIONS.md)、[15项成本合同](../../review/OMISSIONS.md)、[输入hash](../../review/OMISSIONS.md)。

按execution.mode=self，主agent采用第二CPU脚本对账，不称第三方审计。运行时/公共源/旧raw/protected不变；target权重保持9文件size/mtime清单，不冒充全内容hash。另一线程tile派生输出不锁定、不覆盖；Knowledge索引/compact采用保留最新条目的增量写入。历史报告保留，不将旧启动记录或旧失败改写成当前成功结论。
