# 同负载 T/A_off 成本对照设计（草案，未运行）

日期：2026-10-11T13:07:01+08:00（Asia/Shanghai）。仅投机采样 / action-only，非tile。用户本轮同意继续CPU诊断与设计；**没有授权或启动本文件的新GPU任务、同步包装或训练**。

## 1. 把两个问题拆开

- **轨迹问题**：同task/init和受控起点下，S0/A_off是否在第二次抓取额外返工？现有600步与293步来自不同init，不能当T/K2反事实。
- **工作预算问题**：同请求阶段、可比KV长度及实际双worker需求下，A_off是否仍比T每次policy更贵？少env steps但没少chunk，不一定节省policy工作。
- **自然E2E问题**：同步屏障/profile会改变调度与耗时。诊断的逐请求成本不能替代无屏障、profile-off完整episode的实际E2E。

## 2. 源码已具备什么、缺什么

已有默认关闭的WAM_P0_PHASE_PROFILE记录tree/draft/verifier forwards、logical/CFG rows、accepted depth、committed scheduler updates及video/action细项；阶段边界有CUDA同步，因此只作诊断。计数器依赖active profile，不存在独立轻量counter开关。

- T普通循环并没有同名target counter；应读取action_target_forward.calls、action_terminal_target_forward.calls与video_target_forward.calls，不能把空counter当0工作。视频/动作配置20/50个scheduler updates，terminal target调用另算，必须以实际records核验。
- 当前普通K2仍每tree验证7逻辑行；CFG配置video=5/action=1，但use_cfg因video而开启，候选实际14 rows。普通K2不消费leaf，bonus才消费；本设计不改为3行、不启bonus、不改验收阈值。
- infer的结构化profile只覆盖policy，不覆盖reset/cache内部细项；client已有三类累计请求计时。reset/cache保留独立请求账本，内部未知保持未知。
- 现有launcher强制WAM_P0_PHASE_PROFILE=0；profile路径、每请求边界/context与双worker屏障需要**新任务专用包装**，尚未实现。纯CPU的15项静态/调度模型检查不等于真实包装或WAM运行通过。
- 已有counterfactual cache命令能恢复指定记录的部分video/observed-action历史，但不是完整sim/server RNG/hidden state快照，不自动证明同输入成本反事实。此步不启动短截断、one-chunk或新缓存优化。

## 3. 先补最小物理对照，避免误把init难度当算法损失

**建议下一关最多4个完整episode：T/S0 × init3/4。** 每条件2独立EGL worker、算法条件串行；同受控client seed20260922、PYTHONHASHSEED=0、server首局seed20260922与相同公共指纹/原生观察器；target、S0、global threshold、50-step action、20-step video、原cache/CFG/VAE和完整800 horizon不变。teacher/训练/full trace/phase profile关闭。

此处不是成本profile，更不当clean性能表。fresh S0须对上已保存受控轨迹；初态/源/参数或S0复现不通过，先停关卡，不自动追加局数。成功与完整horizon失败都保留。另需新GPU预算与任务包装范围确认；当前只提出，没有运行。

看同init的first soup In、second-object首次双指contact/首次抬升、失败闭合段、success步与chunk，并核对发生关键动作前的观测/输出：

- 若T很快建立第二次抓取而S0复现四次无双指接触：优先定位同状态/同candidate的关键位姿/抓取误差；再定冻结target困难状态监督，不先全面RL。
- 若T也类似长轨迹：先查共同控制/观测/环境起点与任务难度，不将全部600步归因draft。
- 若新S0不能复现：暂停解释新的T差异，定位条件/第一分叉；不“再跑到成功”。

## 4. 成本诊断候选：T/A_off双worker请求队列（8局上限，另授权）

### 条件与预算

- 固定T=mode off；A_off采用夜间冻结末步checkpoint，不重新训练：
  `/rjs/cxz/论文探索/投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt`。
- 2独立EGL worker分别init3/4，各条件首局；条件顺序候选T→A_off→A_off→T，每次重启**本任务自有**server及统一seed，共8完整episode，条件不混跑，失败不扩预算。服务/源码范围与预算执行前另确认。
- 仅诊断profile+请求边界，原生字段可用作离线阶段标注；无teacher/训练/bonus/阈值/cache/算子变更。只能在tile任务自然结束且GPU/互斥锁/端口/EGL门通过后启动，绝不停止并行tile任务。

### 调度与可比性（尚未实现）

1. 将reset、cache、policy分开排队；两个worker都ready后提交同一policy cohort，并记录cohort_id、每个client/server的真实monotonic起止、排队/屏障等待、frame_st_id、实际batch/可读取KV尺寸、prompt/config/source hash。
2. 屏障不得增加/跳过env.step、模型调用或改变动作前缀；需要mock CPU合同和真实完整轨迹验证。它固定**请求到达队列**，不保证GPU kernel同时运行或完全相同资源分配。以真实server区间计算重叠，而不是只看“两进程还活着”。
3. 一个worker终止后解除屏障；另一worker继续原完整horizon，尾部标记single-demand，**不能为补负载额外调用已结束policy，也不能截断未结束episode**。CPU调度模型已检查19/38、19/19、51/19请求保留，真实运行器仍缺失。
4. 跨条件只比较共有frame_st_id/首chunk或稳态/实际并发类别/可比KV形状和离线物理阶段的分层；若两个轨迹同序号chunk但任务阶段不同，不能写成同phase。样本不匹配则标missing/unmatched，不强行校正或选择最快窗口。
5. 纵然阶段、shape和需求匹配，T/A_off的观测/条件历史仍随各自轨迹变化；该实验是**负载受控的闭环成本诊断**，不是同完整隐藏状态的严格模型因果benchmark。需要更严格反事实时另建完整历史/噪声合同，不悄悄用reset+几帧代替。

### 成本表与对账

- client reset/policy/cache累计时长及调用次数；cohort等待单列，不混进policy计算；残差含环境/写日志等，不叫纯env时间。
- server policy总wall与四个顶层项obs_encode/video_total/action_total/postprocess；嵌套项不与action_total重复相加。
- tree/draft/verifier counts、7逻辑/14 CFG rows、accepted_depth分布、普通/terminal target calls；scheduler实际committed与fallback/bonus分类保持边界（bonus本设计0）。
- 分physical stage与frame/KV长度报告request数量、raw分布、匹配覆盖率与single-demand尾部；失败/长尾保留，不仅取success或peer结束后的快请求。
- rows × calls是工作规模代理，**不是精确FLOPs，也不按14/2直接预测7倍时间**；实际网络shape、KV长度与排队影响另报告。

## 5. 独立的profile-off E2E验证（不包含在自动下一关）

若诊断显示成本可解释且候选有继续价值，再另定自然调度、无cohort屏障、profile/trace/teacher/native/per-step fingerprint关闭的完整T/A_off开发集对照。公共低量事件日志仍属计时插桩，记录其开销边界，不称零插桩；不拿profile-on或同步等待的E2E声称部署加速。

success→env steps/长尾→policy chunks/实际target work→E2E按序判读。两个init只是开发诊断，不改已有筛选阈值；正式非劣ε/步数δ和新的held-out方案仍需另定。此文没有证明同成功率、更少步骤、端到端加速或RL有效。

## 6. 当前决定

本轮完成CPU阶段定位及成本可行性草案，**不启动新GPU/服务/训练**。基于“第二罐抓取建立迟迟未发生”这一新增证据，推荐先单独确认4局T/S0受控原生对照，再实现/核验成本包装并另确认8局成本预算；两关不打包成无限自动探索。任何关失败都返回第一分叉/合同检查，不追加实验或直接跳RL/联训。
