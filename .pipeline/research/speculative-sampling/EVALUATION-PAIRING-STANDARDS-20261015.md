# LIBERO / OpenVLA / Spec-VLA 闭环评估中的配对标准（2026-10-15）

## 结论与边界

**事实：**查阅的官方完整 rollout 代码均以预存 task 初态、指定 seed、相同最大步数运行各策略，并按任务成功次数统计成功率；它们没有实现“基线与加速策略的首帧相机像素必须逐位相等”或“两策略实际走过的环境步数必须完全相等”的准入断言。这是对这些具体论文/实现的源码结论，**不是**保证不同初始观测也能作为严格逐局因果配对，更不是允许改动我们已预登记的实验门。[LIBERO `evaluate.py::main`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/evaluate.py)，[OpenVLA `run_libero_eval.py::eval_libero`](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)，[Spec-VLA `run_libero_goal_Spec.py::eval_libero`](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/run_libero_goal_Spec.py)。

**推论：**须区分两种命题：①「在同一基准分布上，各策略闭环成功率如何」是统计质量评估；②「在完全相同的已实现初态/观测/RNG 下，策略改变导致了何种逐局差异」是更强的精确配对或复现证明。第一种不以首帧 RGB 位相等、实际终止步数相等为官方必要条件；第二种应保留这些诊断，并在不相等时拒绝相应的强主张。当前 r2/旧 FAILED 的既定严格门必须照原样执行和记录，不能用文献结论追溯放宽。

## 官方方法核对

| 来源 | 实际固定了什么 | 实际质量统计 | 未见的断言 |
| --- | --- | --- | --- |
| [LIBERO NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) 与[官方 `evaluate.py::main`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/evaluate.py) | 每任务 20 个并行环境；`env.seed(cfg.seed)`；由任务的 `init_states_file` 取 `arange(20) % N` 初态并 `set_init_state`；5 个零动作物理 settle；统一 `cfg.eval.max_steps` 上限。 | 20 局中 `done` 的比例 `success_rate`，另存测试 loss；论文的 FWT/NBT/AUC 都基于成功率，并指出成功率比训练 loss 更能衡量操作策略。 | 没有跨运行或跨策略像素哈希、控制器状态逐位、轨迹步数一致性门。 |
| [OpenVLA 官方 LIBERO 脚本](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)、[环境构造函数](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py)、[README](https://github.com/openvla/openvla/blob/main/README.md) | 默认策略随机 seed 7，`get_libero_env` 对每个任务建环境并 `env.seed(0)`；每 task 使用同一预存 `initial_states[episode_idx]`，每局 `reset`、`set_init_state`、10 个 dummy settle 步；默认每任务 50 局；各 suite 固定动作步上限，`libero_10` 为 520 加 10 settle。 | `done` 成功数 / task 局数、总成功数 / 总局数。README 表格为 3 seeds × 每 seed 500 rollout（10 task × 50），报告均值±误差。 | 没有基线/加速策略初图逐位比较，也没有同一 `episode_idx` 的实际终止步数相等断言。 |
| [Spec-VLA EMNLP 2025 论文](https://aclanthology.org/2025.emnlp-main.1367.pdf)、[AR](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/run_libero_goal_AR.py)、[Spec](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/run_libero_goal_Spec.py)、[Relaxed](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py) | 各 arm 独立运行；默认 seed 7；沿用预存 task 初态与每局 index，构造环境时 `env.seed(0)`，10 个 dummy settle。论文主结果每 task 50 次；**当前公开 Goal 脚本 AR 默认 50 次，Spec/Relaxed 默认 10 次**，直接按默认值跑三脚本不能当论文的同样本量对照；论文消融另有每 task 10 次。 | 论文表 1 报各 suite 的闭环 SR、生成 speedup、每次 forward 预测 token 数 `Length`；代码逐 task/总局统计 `done`，并记录推理调用时间。`Length` **不是**物理环境轨迹长度。 | 论文和三个 Goal 评估入口没有要求跨 arm 相机逐位或终止环境步数相等。 |

关于「同一物理初态」需更加准确：LIBERO [`ControlEnv.set_init_state`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py) 会把保存的 MuJoCo flattened state 设回仿真并重新生成观测；OpenVLA/Spec-VLA 的官方环境工具还特意把环境固定为 seed 0，并在源码注释中提醒 seed **即使用固定初态仍可能影响物体位置**。[OpenVLA 实现](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py)，[Spec-VLA 实现](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/libero_utils.py)。因此「相同 task/init index/seed」是官方可核查的设计配对单位；「完整场景、相机、控制器、所有 RNG 必定逐位相同」并未由该设计或代码证明。这个最后的判断是对源码保证范围的推论。

## 对当前结果的解释和后续报告口径

本项目的 **T10 两次 10/10、总步 2721 与 2748** 可分别描述为该 task/init 集上的成功率和实际终止步数；步数不相等并不违反上述官方成功率评估，因为 `done` 时各局提前结束，且不同 rollout 可以有不同动作路径。但两个 10/10 不能证明真实任务成功率不下降，10 局太少；报告中保留逐 task 成败、成功局步数分布、失败的上限惩罚统计和分母。论文 `Length` 也不能替代这里的实际环境步数。[OpenVLA `done` 即退出](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)，[Spec-VLA 表 1 定义](https://aclanthology.org/2025.emnlp-main.1367.pdf)。

**推论：**task8 的跨轮首相机 hash 不同，意味着不能声称这些轮次具有逐位相同的策略输入，故不能把它们作为我们“严格逐局完全同条件”证明。仍可在明确初态文件、seed、采样程序、版本、初始像素差异以及分母后，把每轮成功率作为描述性基准观测。task2 即使首个 video/action 张量相同，后续实际 command 已分歧，也只能说首步相同，不能声称整条 T 轨迹逐位可复现；仅凭这项观察无法归因是物理、数值、调度还是代码路径。对尚未完成的 r2，继续执行预登记的首像素、sim/controller/RNG 与结果门，失败如实保留，不更改源码或补跑来制造配对。

如果要走向论文级质量对照，先在每个 arm 使用同 suite/task、初态索引、环境 seed、settle 步数、策略权重与完整 horizon，并公开初态差异及 per-task 成功率；之后扩大到充足的独立完整 rollout、报告均值/区间与实际环境步数分布。与 OpenVLA/Spec-VLA 主结果比较时，应匹配 10 task × 50 init、必要时多 seed；其论文/脚本的模型和 horizon 不同，本项目当前 WAM 800 步结果不能冒称 OpenVLA 的 520+10 协议。此段是基于官方评估设计给出的实践推论，不构成某个论文提出的像素容忍阈值或任务成功保证。
