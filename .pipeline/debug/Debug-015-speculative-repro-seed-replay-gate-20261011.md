# 复现与观察器第一关：固定动作回放通过

日期：2026-10-11T12:13:04+08:00（Asia/Shanghai）。范围：投机采样/action-only，非tile。用户同意先做下一步并要求说明探索内容和阶段性结果。本步实际完成CPU源码审查与两独立EGL worker固定动作回放；公共client/model源码、target、draft、验收阈值、scheduler/cache/bonus均未改。

## 1. 当前探索内容与阶段位置

研究目标一直是 **冻结target下，通过投机采样/draft让完整LIBERO任务更快，同时成功率不下降、环境步数和严重长尾不增加**，不是让L2最小或仅最大化接受率。

| 阶段 | 已有结果 | 当前边界 |
|---|---|---|
| 单chunk工程加速 | 历史稳态policy约6.2772→4.6559秒，约降25.83% | 不等于每种线上配置或整局都加速 |
| 完整闭环性能 | 历史六init配对均成功，但K2环境步数+9.37%、E2E+0.89% | 联合收益尚未证明 |
| 冻结target最小训练对照 | A_off少27步却同72chunk、E2E+9.67%；B_on仅3/4成功 | 未晋级，B_on只是S0访问状态的一轮teacher监督，不是RL |
| 六局原生诊断 | 6/6成功，命令/native/terminal记录完整 | 旧两次失败未复现，不能反推训练改善或根因 |
| **本步复现/旁路关卡** | 发现client seed合同缺口；固定动作重复/开关回放通过 | 完整模型闭环仍待验证 |

本步是在排除测量/复现混杂，确保下一次环境步数差异值得用于训练，而不是转向tile算子优化。L2保留为verifier局部误差，仍不作为任务进度或晋级代理。当前没有RL、target权重更新或真实大小模型联训。

## 2. 新查明的源码事实

1. 正式`p0_libero_multichunk_client.py`记录`LINGBOT_SEED`，但本文件没有`random.seed`、`np.random.seed`或`env.seed`调用；`_init_env`为reset→set_init_state→5步稳定化，没有显式client种子合同。服务器另一个进程的torch/NumPy seed不能自动给client的随机状态设种子。
2. 旧16步`p0_native_binding_check.py`则显式调用三种client seed；因此它的seed合同与正式client不同，不能用它保证原正式运行起点严格一致。
3. LIBERO`set_init_state`进入`regenerate_obs_from_state`，flattened state仅包含time/qpos/qvel；MuJoCo绑定的恢复只写这三者。不包含qacc_warmstart、ctrl、controller goal、RNG、观测/cache等完整隐藏状态。**恢复同一init文件不是完整运行状态回放**。
4. provider审查中未发现step/reset/forward/seed调用；pose读取copy后序列化，contact/predicate/success读取需动态验证。本步12项窄AST语义检查通过，不把它冒充全程序纯函数证明。

这些是可复现合同缺口与风险，不是已证实“未设种子导致旧失败”或“warmstart是根因”。默认生产行为未改；本次回放仅在独立实验进程显式设种子。

## 3. 实际运行与结果

- 12:02:03启动，12:03:52结束，约1.82分钟；分析验证随后完成。GPU只用于EGL渲染，**0模型forward、0新增自适应policy episode、0训练更新**。GPU运行前后显存0MiB。
- 固定上轮S0保存的完整执行动作：task0/init3的298条、init4的310条。两独立worker每个条件并行，OFF-A→OFF-B→ON三个条件串行；每条件新进程，Python/NumPy/环境seed=20260922在构造和reset前明确应用，PYTHONHASHSEED=0。
- 不调用model，不改变固定动作，不补零动作或人为延长。实际三轮两init均在最后保存动作达到done；若fixture耗尽未done将如实标记，而不冒充完整policy评测。
- NVIDIA EGL实际加载、源码digest、init文件hash、model XML hash、worker环境、端口“不适用”（没有服务）与source-action digest都有记录；使用同一GPU互斥lock。写入临时独立输出，不删除或覆盖历史产物。

| init | OFF-A / OFF-B / ON 动作数 | OFF/OFF重复 | OFF/ON开关 | 停止结果 |
|---|---|---|---|---|
| 3 | 298 / 298 / 298 | 测量状态数组、RGB hash、client RNG均一致 | 同左 | 三轮均done |
| 4 | 310 / 310 / 310 | 测量状态数组、RGB hash、client RNG均一致 | 同左 | 三轮均done |

- 共1824次动作（608×3），ON含初始行共610次capture前后检查。每次capture前后20种测量状态数组、Python/NumPy RNG与返回RGB不变。
- 20种状态包括flattened state、qacc_warmstart、ctrl、act、qacc、applied forces、mocap、qpos/qvel、body/site pose及5种controller字段；三条件逐步保存NPZ并逐值比较。每步两个camera的数组hash一致。
- 独立CPU验证36600个状态数组hash（1830行×20字段），重算两份终止NPZ翻转后的RGB hash；三条件四组对比全部通过。2953个旧runtime/raw Knowledge/tile/protected文件和11个审查源码hash不变。

## 4. 可以和不能推出什么

**支持的结论：** 在明确固定client种子、相同保存动作和这两个初始化的条件下，OFF重复基线稳定；未观察到observer capture对已测sim/controller/RNG/RGB的扰动。比原16个无搬运脚本动作更贴近实际抓取、搬运和成功阶段。

**尚不能推出：** 完整WAM数值等价；旧init3/4失败的唯一原因；历史未记录client RNG的精确重建；完整隐藏状态恢复；训练有效性或成功率非劣。这里的动作由fixture固定，不由模型在新观测上自适应重规划，不能替代完整闭环开关对照。

没有直接修复生产seed，也没有强制清零warmstart/controller等改变动力学。GPU仅EGL，不按1.82分钟给policy运行速度结论。

## 5. 排序假设与反证

| 排序 | 假设 | 检查结果 | 状态 |
|---|---|---|---|
| 1 | 正式运行起点并没有完整固定 | 本文件无client seed，init只恢复time/qpos/qvel；旧init3还有前序server RNG历史差异 | 合同缺口明确，旧失败因果仍未证 |
| 2 | provider读数直接修改仿真/RNG | 610次读前后20字段/RNG/RGB不变；OFF/ON整条fixture一致 | 测试范围内不支持直接扰动；非全程序证明 |
| 3 | 同动作下物理/渲染仍随机漂移 | seeded OFF/OFF两init测量数组/RGB hash一致 | 当前固定回放中未复现漂移 |
| 4 | 模型采样/缓存/并行数值效应造成完整闭环分叉 | 本步未加载模型，上一轮init4观测先于命令分叉 | 下一关，当前无法判断 |
| 5 | B_on已经改善了任务 | 本步只回放S0，不更新或比较candidate任务能力 | 没有新增支持证据 |

当前execution.mode=self，主agent完成检查并另编CPU复核脚本；未派第三方独立复现agent，原因是自执行合同及本步窄边界，不把复核称为第三方审计。不修改旧tile state.phase。

## 6. 下一步：完整模型闭环最小关卡（用户已明确确认）

建议最多6完整episode，**同一S0+冻结target**：OFF-A、OFF-B、ON，每条件两个独立EGL worker分别init3/4，算法/开关条件串行，各worker首局，seed=20260922。无teacher、训练/RL、bonus/gate/阈值或模型权重变更。

- 只新增本次job专用种子/公共状态指纹包装：constructor及reset前种子明确应用；所有条件一致记录初始/逐步状态与RNG/RGB hash；仅ON调用native observer。公共model/client源码及默认行为不改。
- server新进程/同startup种子/同模型配置，并保存动作/video张量；完整task到native done或原800 horizon，不限制chunk预算或人为提前截断。
- 先检查OFF-A/OFF-B重复；再解释ON相对OFF偏差。若OFF/OFF不稳定，不用ON差异归因observer；有第一分叉即记录，仍不能为观察差异擅自增加episode或改物理。异常检查停止自有子进程、保留已有数据，无自动重试。
- 比较初态和逐步测量字段、client RNG、RGB、实际动作、缓存边界与输出张量；必要完整server RNG/hidden cache快照未实现，不声称已完全恢复。所有条件共同指纹记录是诊断插桩，不是clean timing。
- 预计25–45分钟，若出现796步长尾可能更久；只估计，不截断。用户随后明确回复“确认，按6局上限执行”，授权此关卡及job-only包装；实际启动状态与时间见下方本任务记录，不把运行中写成完成。

之后才单独做同phase/同需求T/A_off成本计数，再决定冻结target的任务反馈训练。暂不直接RL/KL或大小模型联训。

## 7. 证据入口

- [源码审查](../../review/OMISSIONS.md)
- [回放验证](../../review/OMISSIONS.md)、[独立CPU复核](../../review/OMISSIONS.md)
- [条件对比](../../review/OMISSIONS.md)、[逐字段差异](../../review/OMISSIONS.md)
- [研究持续入口](../research/speculative-sampling/RESEARCH.md)、[前一步结果](Debug-014-speculative-native-six-results-reproduction-gate-20261011.md)

## 8. 用户确认后实际后台启动

启动：2026-10-11T12:10:35.560799+08:00；本报告写入时状态：running_repeat_off_A，supervisor PID 2120659。

13项既有CPU回归与16项新包装scripted合同通过（零policy episode），固定回放PASS后才启动；公共源码/模型默认行为不改，只新增本任务job包装。模型加载和6局尚在执行，不提前报成功或数值等价。

OFF/OFF基线任一已测字段、命令、RNG/RGB或终止长度不一致会在四局后停于基线关卡，不继续ON；原始差异保留，另行决定，不自动扩预算。ON两局如完成则输出自动对比，随后仍需CPU检查保存的完整action/video张量。

- [本次运行状态](../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/status.json)、[固定范围计划](../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/plan.json)、[主日志](../../review/OMISSIONS.md)。
- [自动结果入口](../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/FINAL_REPORT.md)指向任务FINAL_REPORT，任务结束或失败自动生成。
- 服务器独立tmux spec-repro6-20261011-120724/repro6，PTY stdin保留、子进程/dev/null；不依赖本机屏幕/SSH。服务器重启/硬件故障不保证。停任务使用本任务STOP_REQUESTED文件，由supervisor只清理自有子进程。
- 运行25–45分钟是估计，长尾或关卡停止均可能改变。下一步：读OFF/OFF及ON/OFF第一分叉，完成CPU张量/观测核对；之后才同负载成本复测，无自动训练/RL或联训。

### 运行来源标签补充

复用旧launcher时，OFF条件的原始run_metadata保留latency_benchmark=true及原client脚本路径；这只是旧启动标记，不是本轮clean timing资格。真实/proc命令已验证使用job-only client_repro.py，repro_metadata与worker_launch_config明确common指纹插桩、clean_timing=false。本轮全部条件均按诊断处理，不据这个legacy标记计算性能收益；原始metadata不覆盖、不静默改写。
