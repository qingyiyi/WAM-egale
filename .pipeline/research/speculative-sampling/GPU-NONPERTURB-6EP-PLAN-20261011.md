# 最小真实GPU不扰动验证方案：最多6新完整episode（待确认）

日期：2026-10-11T15:07:45+08:00（Asia/Shanghai）。**投机采样action-only，非tile。只规划/读已有数据/记录HW：未获本方案GPU预算，未启动GPU、模型服务、teacher或训练。** 不修改当前成本包装、公共模型源码、权重、旧runtime/raw或protected文件，不新建Cycle/Milestone。

## 1. 问题与最小预算

本关只验证：加入**profile＋成本sidecar＋双worker cohort**这一组合后，在冻结配置及两个受控init上，模型完整输出和LIBERO已测完整轨迹是否仍与关闭时精确相同。不是验证训练收益/成功率非劣，也不是自然E2E性能。

建议批准**最多6次新episode尝试，目标6个完整episode**：3轮×2个独立EGL worker，worker0/init3、worker1/init4，各server/client均首局；轮次串行，不混跑。启动后失败也消耗名额，不自动重试/补局。

| 顺序 | 条件 | profile/成本包装/cohort | 新完整episode数 | 比较对象与闸门 |
|---|---|---|---:|---|
| 1 | T_ON（target-only） | 全开 | 2 | 复用已完成的同种子/首局/native+指纹T_OFF参考；两init均通过完整对账，才进第2轮 |
| 2 | A_off_OFF（冻结小模型＋target/verifier） | 全关 | 2 | 新建同种子、首局、公共native+指纹的参考；完整性及初态合同通过，才进第3轮 |
| 3 | A_off_ON（同一冻结小模型） | 全开 | 2 | 对比本任务A_off_OFF的全部输出与已测轨迹；结束后CPU总验收/写HW，停止本关 |

“ON/OFF”只指上述**诊断组合**，不指算法模式开关；A_off两轮算法都仍ordinary `verifier_k2_b2`，T仍`off`。native/公共fingerprint两侧都开，teacher/full trace都关。不同算法不要求互相输出相同，只比较**各自**ON/OFF。

### 为什么6，而不是4或8

- 已有可靠T_OFF在 `/rjs/cxz/论文探索/投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403`：init3/4成功293/300步、各19 chunks；20初始数组/client RNG/RGB、595行native/指纹、76份完整action/video张量及terminal均保留。公共source/权重/种子/首局/插桩合同当前可核对，因此可**有条件只补T_ON两局**。
- 旧A_off夜间eval没有当前受控client seed/native/公共指纹，且init3是worker第二局；即使init4首局，也缺全测轨迹证据。旧287/289步不能写成新合同期望。必须新OFF2＋ON2共4局。
- 只跑4局仅覆盖A_off，不覆盖新增插桩在真实T路径的行为。每条件要求双独立EGL，所以不能用一局T来补。
- 若要求所有OFF参考均现跑，需8局（T_OFF2＋T_ON2＋A_off_OFF2＋A_off_ON2），预计比6局多一轮约8–15分钟。**若旧T来源合同或复现不符合，当前6局方案直接停止，不自动扩成8局**；新预算再确认。
- 6局是复用可靠参考的最小工程关卡，不是统计学非劣性样本量、全任务/全hidden state证明。不开OFF/OFF追加重复；出现复现疑点则另行设计，不“跑到一致”。

## 2. 时间与安全预算（估计，不保证）

当前只读的历史运行证据：受控T双worker每局约374秒（6.24分钟），其中reset约127秒；旧A_off各局369–388秒（6.16–6.47分钟）；已有S0长尾600步双worker轮约9.6分钟，之前6局三轮复现总31.03分钟。以上用于估时，**不把不同seed/算法的旧步数或时长当新A_off结果**。

- **GPU运行阶段预计30–50分钟**：包括三轮的本任务服务启动、reset、真实完整episode、清理及轮间必要关卡；profile同步和cohort等待可能增加时间。
- **准备＋GPU＋CPU逐元素核验＋HW汇总合计45–75分钟**；不含等待其他GPU任务自然完成。新的A_off受控首局/长尾尚未知，可能更久。
- 建议同预算批准**GPU运行阶段90分钟安全上限**，从首个本任务server启动计，不含GPU资源排队和前置CPU准备。到限中止自身任务，记failed/incomplete并保存已有证据；这不是“截短episode后当通过”。不重试或扩大名额，不擅自延长。
- GPU未在本轮执行resource gate，不声称当前空闲；仅CPU磁盘stat当时约61.9GiB。启动前重新核验共享独占锁、实际GPU/显存/进程、磁盘（原40GiB门与运行12GiB停止线）、server/master/broker端口、EGL及provenance，不停止/争用tile任务。

## 3. 冻结合同

- `libero_10`、task0；init3/4，各条件同worker映射、各自fresh server/client首局，不resume，不混历史RNG。
- server和client seed20260922、PYTHONHASHSEED=0；同公共seed/native+fingerprint包装，初始20已测数组、model XML/init-state hash和双RGB/client RNG必须对齐。
- target公开Long checkpoint；A_off固定夜间末步 `/rjs/cxz/论文探索/投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt`，验证原内容SHA，不再训练。
- selector global、threshold 0.007839680526450805；original cache/torch attention、CFG/video5/action1、50 action updates＋terminal、20 video updates＋terminal、VAE/offload、动作前缀/stride/成功检查时机不改。
- 完整simulator horizon800，5稳定步；不人为max-policy-chunks或one-chunk截断。未成功则走原完整horizon（通常795计入实验的env steps），不把800去噪步/环境步混为一谈。
- teacher/full trace/RL/KL训练/大小模型联训/bonus全关，不更改L2验收阈值，不借GPU验证修算法或cache/算子。

## 4. 通过标准：五组必须全部满足

### A. 来源及初态

fresh source/config/input hashes、target size/mtime、A_off内容hash、旧T原产物hash、组件/环境及EGL合同有效；同init/worker/server首局/种子。双方20初始数组shape/dtype/值、client RNG、XML/init-state与两视角初始RGB相同。若变更或缺字段，停止，不将missing视作相等。

当前规划只核查了761个冻结文件hash、248预备source/input、9target size/mtime及旧T资料存在/连续性；**还未做新的89项CPU回归，也没有新的GPU验证**。上一关89 PASS只作已有工程基础。

### B. 完整模型输出

每一个对应policy chunk，完整30通道action与video latent载入CPU后shape/dtype一致，`torch.equal`逐元素精确相同，不用allclose、平均L2、只看执行的7通道或只比较少数关键帧代替。NaN/Inf、缺文件、重复/少chunk一律失败。若对外称“逐位相同”，补CPU张量数值字节hash，不要求torch.save序列化文件字节相同。

- T_ON必须匹配旧T：init3 293步、init4 300步；各19 chunks、共76份完整action/video比较。
- A_off_ON必须匹配**新A_off_OFF实际记录**，不预先指定287/289/600等旧步数。

### C. 已测完整闭环轨迹

逐env.step的已测state hashes/client RNG/双RGB/实际命令/done/reward，原生对象位姿/contact/goal terms/gripper/EEF、simulator timestep，以及终止原始双RGB、成功/horizon、环境步数、chunk数均精确相同；native/fingerprint行数=env_steps＋1、连续无errors。wall-clock、时间戳、PID、路径、condition/request id等诊断标识不要求相同。

同样的最终success或总步数**不够**；任一早期输出/状态/RGB分叉都不通过。A_off两侧若均相同地完整horizon失败，则可以通过“不扰动”这一单项，但必须报告候选任务失败，不能据此晋级或声称准确率达标。

此标准是全部**已测**字段/完整输出的操作性不扰动，不声称未保存的server RNG、所有隐藏KV、所有内部physics微步已完全恢复或全等。

### D. 调度与请求账本

ON所有reset/policy/cache的client-server request-id/metadata一一对应、序号连续，真实起止时间同boot时钟合法；两活跃worker ready才release，正常结束后仅剩余worker继续完整尾部，planned demand降到1。不多/少请求、不改env.step，不补假peer、不抛弃尾部；timeout/重复/错序/worker错误失败。记录真实server overlap，**不要求它一定为正，也不将它当GPU kernel重叠**。

若两个新轨迹恰好同chunk数、未出现single-demand尾部，记录GPU尾部分支未覆盖，不能额外加局强造长尾；已有CPU尾部合同不冒充GPU覆盖。

### E. 原始计数与真实网络shape

profile与sidecar一致、每actual transformer forward覆盖所有attention blocks，读取有效KV选择后的QKV shape/CFG rows；T每policy普通action50＋terminal1/video21。ordinary K2七逻辑/十四CFG行，实际calls与原counters、50 committed scheduler updates、fallback及terminal对账，无bonus/额外forward；fallback提交不自动记成另一次target forward。缺sidecar/phase字段不补0，顶层与嵌套时间不重复加。

reset/cache内部phase缺失是已知边界，保留未知，但请求区间/实际forward若发生仍须完整记录。A_off_OFF未开profile，不能伪造OFF的同名counter；通过全输出/轨迹与ON内部账本合同联合验证。

## 5. 失败规则与结论解释

- 第一轮T_ON任一init不匹配：本关最多用了2局，停止，不运行A_off。第二轮A_off_OFF完整性/初态失效：最多4局，停止，不运行ON。第三轮ON任一不匹配或账本失效：总最多6局，记录FAIL/第一分叉，停止。
- 任何OOM、异常、EGL/端口/来源失配、安全超时、资源争用/磁盘不足都保留已完成/失败原始证据；只清理本任务pid组，不删除旧文件，不补局、不改阈值/容忍度“跑通”。
- FAIL只能说明当前合同/复现/诊断组合不能继续作可靠成本对照，**不能直接判定是profile、cohort、某个wrapper或模型训练的唯一原因**；三者本轮一起切换，拆因子另授权。
- PASS只允许判定这些冻结配置/两个init的**组合诊断协议**满足已测不扰动标准；不要求时长相同或更快，profile同步/屏障本来会改变时长/调度，不证明clean E2E、GPU kernel重叠、成功率非劣或多任务泛化。

## 6. 执行前还需补的工程（本轮未实施）

目前已有成本`campaign.py`是受保护的**8局成本入口**，要求GPU不扰动先PASS；不能用它跑本关，更不能先把PASS写死来绕过。

预算确认后，仅新增本任务**6局专用监督器与通用CPU ON/OFF比较器**，复用现有默认关闭client/server成本包装与旧seed包装；OFF使用原server＋同seed/native/fingerprint client，ON使用当前cost entries。不改公共源码、原S0专用比较器或8局成本guard。补CPU范围/预算/模式/失败/尾部、旧T自比较及故障注入；自比较应76张输出/595行全等、故障注入正确拒绝；fresh CPU不过不占GPU。

之后才资源/EGL/provenance门与服务器后台启动。长任务使用服务器独立tmux/PTY、自有监督器/超时/原始数据落盘，不依赖本机SSH会话保持；不保证服务器重启/硬件故障继续存活。

## 7. 每一步汇报与后续边界

预算确认前停在本文件。若用户确认“6局＋90分钟安全上限＋必要任务专用监督器/CPU比较器”，按关卡在此预算内推进，**不再扩GPU任务/训练**：

1. CPU准备完成：记录HW、说明是否可进真实GPU第1轮。
2. T_ON两局/旧T复现完成：报告PASS/首分叉，提醒下一步仅A_off_OFF两局；未通过就停。
3. A_off_OFF两局完成：报告新基线实际success/步数/chunks，不当改善结论；提醒下一步A_off_ON两局。
4. A_off_ON及CPU总验收完成：报告各字段/全部张量/账本/覆盖/失败，写HW、Knowledge/index/compact/PROGRESS，并提醒下一步。
5. 全PASS后，**最多8局T/A_off同负载成本诊断仍需另外确认**，不会自动开跑。自然调度/profile-off/无屏障完整E2E再独立确认；困难阶段同candidate冻结target监督训练继续后置，L2辅助，任务成功/步数/长尾/work/E2E主验收。

当前结论：**6局方案已规划，GPU批准=false，真实验证=NOT_RUN，训练未开始；等待预算确认。**
