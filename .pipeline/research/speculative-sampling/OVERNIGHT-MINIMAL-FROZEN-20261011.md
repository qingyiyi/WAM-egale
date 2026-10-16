# 最小冻结 target 夜间对照：启动与预登记

记录：2026-10-11T00:49:03+08:00；只属投机采样，非tile。

## 决策纠偏与当前状态

100episode纯基线方案有描述性价值，但不能验证新训练方法，用户指出偏离最小对照后已在GPU启动前撤下。用户明确授权必要源码/训练改动与最多20完整episode、自动按验证关卡推进；随后明确无需再等计划确认。

当前：running_collect_T，supervisor PID 1247023；训练和最终评测尚未完成，不将预测写成结果。

## 已通过的前置关卡

- 默认关闭client native observer：稳定化后初始状态+每动作状态，copy姿态/qpos、同grip-site位姿、双物体双指接触、In/native success、terminal RGB；特权状态不进入模型。
- 37 CPU合同/回归 tests、9真实WAM模块CPU unittest通过，服务器模块可导入；首次试用错误环境缺flash_attn/VA缺pytest已改用现有正确环境及unittest，无依赖安装。
- 真实LIBERO provider脚本：observer off/on各16规定动作，模拟器轨迹一致；读取不变sim/Python/NumPy RNG，字段及两项goal合取与native success一致。它不是完整policy评测，也未证明所有WAM真实数值on/off等价或持稳物理标签有效。
- 新action-only teacher旁路实现保存每棵K2树3个实际predictor输入(root与两条internal)、实际target CFG标签、previous方向、sigma/next_sigma/timestep和query hash；包含非选中branch，不只fallback。每chunk attempted/saved/3node marker对账；无额外target forward。真实数据是否完整要等collect_S0关卡。
- 当前部署B2 draft继续训练接口已通过CPU两次更新/loader/normalization测试；历史task0/init0实际x_i数据CPU可读833条、17文件。真实A_off/B_on GPU训练尚未执行。

## 固定实验与预算

| 阶段 | 内容 | policy episode数 |
|---|---|---|
| 采集/诊断 | T和S0，各init0/1；仅S0采同query teacher | 4 |
| 训练 | A_off历史target-only task0/init0；B_on固定S0真实访问状态teacher；相同原B2初始化、归一化、架构、loss、optimizer | 0 |
| 干净评测 | T、S0、A_off、B_on，各init2–5 | 16 |

总上限20完整episode；不自动重试/扩预算。GPU上算法臂串行，每臂2独立EGL worker，seed20260922，libero_10/task0，horizon800含5稳定化、policy步数排除稳定化；不人为缩短评测。target robbyant/lingbot-va-posttrain-libero-long，现有draft/global阈值 0.007839680526450805；不改target/verifier/gate/bonus。

每臂1000 optimizer更新、batch512、lr1e-4；唯一数据预算n=min(A_off可用,B_on可用,4096)，共同n≥128才训练，抽样次数相同。两臂保存固定末次checkpoint，不按评测挑最好模型。A_off实际x_i而非未经准备的前步raw x_next；CFG/action/cache合同一致性另检。B_on为S0访问分布上的**单轮监督更新**，仍跟真实teacher，不是RL或任务reward优化，不声称最终candidate自采样on-policy。

成功率/失败→配对完整步数与长尾→E2E为主；L2只是训练/解释，不直接晋级。所有失败全报告并使用公共795策略步horizon惩罚，实际步数/耗时也保存。pilot筛查参考：无更多失败、均值步数≤T×1.05、max步数≤T×1.10、E2E优于S0；只是启发式，不是正式统计非劣。init2–5此前已看，历史训练暴露未排除，仅本轮更新隔离development。

## 预期是假设，实际结果决定后续

- 预测：B_on更可能改善访问状态上的局部teacher拟合，但环境步数/掉落/E2E是否改善不确定；不能承诺同准确率下更快。
- 若B_on在任务和E2E方向一致优于A_off/S0：只成为扩大多任务/预锁容差评测的候选，不直接进RL/联训。
- 若A_off与B_on近似：更像额外训练收益，不能归因到闭环分布；检查覆盖、预算和任务瓶颈。
- 若L2改善但任务/步数无改善：支持继续设计基于native诊断的任务事件/反馈目标，而非盲目延长teacher训练。
- 若成功率恶化、数据/接口不完整：拒绝晋级，停在失败关卡，留证据和下一步修复方案。

## 后台与查看

独立服务器tmux socket spec-minimal-20261011 / session frozen20；supervisor父进程为tmux daemon（daemon PPID1），不依赖本机屏幕、SSH或Codex继续运行。保留tmux stdin防PTY全部关闭导致SIGHUP，stdout/stderr到日志；子任务stdin=/dev/null。首两次无GPU启动因PTY关闭结束，已修复。不能保证服务器断电/重启、OOM或磁盘硬件故障仍继续。

预计2–4小时，只是估计；失败full horizon长尾可延长，不为凑时长跑100基线。40GiB阶段门、12GiB紧急停，source hash和target9文件stat清单固定（**不是全权重content digest**），只清理自有子进程。

- 运行目录：[job](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242)
- 状态：[status.json](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/status.json)；主日志：[supervisor.log](../../../review/OMISSIONS.md)
- 运行计划：[plan.json](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/plan.json)；完成/失败后自动写：[FINAL_REPORT.md](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)
- HW动态结果入口：[RESULT](../../../投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/FINAL_REPORT.md)，symlink指向该任务自动报告，旧raw记录不回写。
- 停止：服务器执行 touch '/rjs/cxz/论文探索/投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/STOP_REQUESTED'；不要只关闭tmux，以免绕过自有子进程清理。

**下一步正在自动推进：4episode采集→完整数据门→同预算训练→16episode干净评测→按真实结果写下一步。明早先看failed/completed及报告，不直接开RL/真实联训。**
