# 投机采样固定负载成本对照：第一次启动止步与 CPU 复核

2026-10-11（Asia/Shanghai）；**投机采样，非 tile**。

## 当前结论

用户确认“不重构，先成本对照”。本轮已补齐任务专用、默认关闭的包装、快照恢复、双 worker 同步、独立 CPU 比较器、预算/限时监督器和逐关 HW 记录，并通过 CPU 工程预检。

**GPU 第一次启动在两模型加载后的任务包装初始化处失败，已经停止并清理；未进入 EGL、fixture 恢复、不扰动或四格成本主测。** 没有新成本数据，不能据此判断投机方案快慢、KV 根因、任务成功率或环境步数。

这是新增包装的初始化顺序错误，也是 CPU 预检覆盖遗漏；不是模型质量失败、显存不足或成本退化。历史六局和已有成本账本结论保持，不用本次工程失败推翻“不重构、成本优先”的研究决定。

## 已完成与未完成

| 步骤 | 结果 |
|---|---|
| 用户授权与 HW 记录 | 完成：0 新完整 episode、最多 2 次 EGL 初态、168 完整固定 chunk、90 分钟 GPU 硬限时，失败停止，不训练 |
| 任务专用实现与 CPU 预检 | 195 项测试通过：新固定负载 63、旧成本包装 76、旧不扰动 43、最小冻结 target 13；实际测试进程退出码 0 |
| CPU 输入与冻结边界 | 84 个真实旧产物、440 个双相机窗口帧；1128 个旧文件 SHA、9 个 target 文件 size/mtime 核查通过；CPU gate 未初始化 CUDA |
| GPU 启动前资源 | 通过：GPU 显存 0、无 compute app，内存/磁盘/端口检查通过；服务器 tmux 独立启动 |
| 两模型 worker 就绪 | **失败**：两边都在任务 Runtime 的 draft 预加载处触发原 API 的模式保护；监督器停止 |
| EGL / 初态一致性 | NOT_RUN；初始化 0/2，稳定步 0，policy 环境动作 0 |
| Prefix / 四份 fixture / 精确恢复 | NOT_RUN；prefix forward 0 |
| 包装 OFF/ON / 单双需求不扰动 G1 | NOT_RUN；固定 chunk 0/168 |
| S1 → L2 → L1 → S2 主测与 FINAL | NOT_RUN；无 action 成本比值 |
| 失败后 CPU 复核 | 完成：复现真实 API 保护与测试盲点；没有模型/Torch 导入或 CUDA 初始化，没有修改冻结实现或重启 |

原计划状态 `CPU_ENGINEERING_PASS_GPU_NOT_RUN` 是启动前的冻结快照，不再写改；**本次实际执行终态以 `status.json`、`FAILED.json` 和本报告为准：FAIL，已清理。** 原待确认提案也保留历史状态，不覆盖。

## 失败根因与 CPU 预检盲点

1. 监督器按默认关闭合同启动模型：`WAM_P0_ACTION_DRAFT_MODE=off`，新建原 server 的 predictor 尚未加载。
2. 新 `Runtime.__init__` 直接调用 `_load_p0_action_draft_b2_predictor()`。
3. 原 API 在 predictor 不存在、模式不为 `verifier_k2_b2` 等支持值时，抛出 `B=2 verifier draft requested while mode is disabled`。两 worker 日志都给出相同调用栈。
4. CPU 包装测试预先填入了假 predictor，并把 loader 替换为 lambda，因此绕过了这个真实保护检查。195 项通过真实有效，但不足以证明首次加载的集成合同。

失败后 CPU 复核从**未修改的真实公共 loader 与完整任务 Runtime 类**抽取 AST，分别构造 fresh 与 cached predictor 的 CPU 对象：fresh/off 确切复现同一异常；cached/off 提前返回而完成包装初始化。这里没有导入模型或 Torch、没有真实加载权重、没有 GPU。该探针证明保护检查/初始化顺序矛盾，**不证明修复后的真实模型或后续恢复关可通过**。

## 实际消耗、清理与边界

- 服务器 tmux 启动：约 18:13:34；GPU 安全计时开始：约 18:13:39.73。失败在 18:14:11 左右被检测；监督器失败快照记录约 **32.06 秒**，18:14:13 已写出终态。
- 两次模型 worker 启动，**不是两次 EGL 初始化**。新完整 episode **0**、EGL **0/2**、固定完整 chunk **0/168**、prefix forward **0**；没有重试、补局或额外测量。
- 只清理本任务拥有的两个进程组；`OWNERSHIP-CLEANUP.json` 为 PASS，残留 owned PID 和监听端口均为空。随后 `nvidia-smi` 显存 0 MiB、compute app 为空；tmux 已退出。
- 90 分钟独立 watchdog 未触发；本次远低于限时，不声称本次短启动已实测验证 90 分钟故障路径或 SSH 断连。
- 未训练、采 teacher、RL 或大小模型联训，未修改公共 WAM/server/client、旧 guard、原验收阈值、bonus、target 或 A_off 权重；11 个新实现文件/计划 SHA、1128 个旧文件 SHA、9 个 target stat 写后复核一致。protected state/rules 未修改；不改 tile 研究条目，也不回滚其他会话的共享 HW 更新。

## 下一步：先 CPU 修复，GPU 再单独确认

**建议，不是已经执行：**

1. 只修任务专用预加载顺序：在未计时的初始化区临时使用原 `verifier_k2_b2` 模式加载实际 A_off predictor，以 `try/finally` 恢复原 `off`；不删除原 API 保护、不修改公共默认行为。种子/fixture 捕获仍在初始化之后。
2. 增加使用真实 loader 合同的 CPU 回归：fresh/cached predictor、默认 off、成功及异常恢复、正确 checkpoint/几何和 device 路径；保留现有故障/不扰动测试。先明确验证真实 A_off 在 CPU 的加载合同，再考虑 GPU。
3. **修复须另确认；本轮不修改冻结实现，不恢复此 one-shot 失败 job。** CPU 修复通过后，如用户确认再次 GPU，建立新 job、重新冻结实现与 CPU gate、重新检查资源；仍按 0 完整 episode / 2 EGL / 168 chunk / 90 分钟和失败停止执行。未用完额度不等于自动续跑授权。
4. 新启动通过后才继续：初态 → prefix/全缓存恢复 → G1 不扰动 → 四格成本。成本关做完并 CPU 汇总后，再讨论成本感知启停、世界条件 draft 或冻结 target 真实状态训练；训练/RL/联训均不自动启动。

## 证据

- [原始任务目录](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601)
- [195 项 CPU gate](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/preflight/cpu-gate.json)；[实际测试日志](../../../review/OMISSIONS.md)
- [执行失败](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/FAILED.json)；[worker0 调用栈](../../../review/OMISSIONS.md)；[worker1 调用栈](../../../review/OMISSIONS.md)
- [owned 清理审计](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/OWNERSHIP-CLEANUP.json)
- [失败后 CPU 根因复核](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/CPU-FAILURE-READOUT.json)
- [写后核验](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-174601/POST-RECORDING-AUDIT.json)：1244 项 SHA/stat/记录/索引/最新写入保留与终态检查通过；执行终态仍为 FAIL。compact 是派生摘要，不要求历史每个事件保留四条独立普通 bullet，canonical records 与索引均已核验。
- 逐关 AUTH / CPU / START / FAILED 报告与 Knowledge 记录及 POST-CPU 根因记录均已保存，不改写 FAIL。
