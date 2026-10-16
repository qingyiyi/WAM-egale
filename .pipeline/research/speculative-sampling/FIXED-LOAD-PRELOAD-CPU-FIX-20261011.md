# 投机采样：任务预加载顺序修复与真实接口 CPU 回归

2026-10-11（Asia/Shanghai）；**CPU 修复通过，GPU 未获本轮启动授权，非 tile**。

## 结论与范围

按用户确认，本轮只修任务包装的预加载顺序，并补真实接口 CPU 回归。不重构、不重启 GPU、不训练。

**预加载顺序错误已修复，209 项合并 CPU 回归和原 `va` 环境的真实 server 接口 CPU 探针均通过。** 这只证明本次初始化合同及 CPU 异常恢复；GPU cache 恢复、OFF/ON 不扰动、单双需求与四格成本仍为 NOT_RUN，没有新成本/成功率/环境步数结论。

## 最小代码变更

只修改两个已有任务文件：

- [worker.py](../../../review/OMISSIONS.md)：新增 `preload_action_draft(server)`，在未计时的初始化区临时置原接口支持的 `verifier_k2_b2`，通过原 loader 加载实际 predictor 并 `.to(server.device).eval()`；`try/finally` 在成功和加载/迁移/eval 异常时都恢复调用前模式。Runtime 用它替换旧的直接预加载。
- [test_cpu.py](../../../review/OMISSIONS.md)：原合成 infer 测试补明确 `off` 初态；新增 14 个真实加载合同回归。未改原公共 loader 的保护、checkpoint schema 或原预测算法。

原 Runtime 除 constructor 的预加载替换外，其余全部方法 AST 完全相同；`build`、`run`、计时/同步、cache/RNG 捕获恢复、原推理 loop/验收未改。其他 9 个任务实现文件保持旧 SHA。公共 WAM/server/client、旧 guard、target/A_off、阈值、bonus 和 protected state/rules 均不修改。

模式恢复不是 predictor/RNG 全状态回滚：原 loader 若成功缓存 predictor 后设备迁移失败，该对象仍可能存在；包装不隐藏异常、不重试、不发布 ready。真实 MLP 构造按原实现可能消耗 CPU RNG，原 fixture 的固定 seed 捕获仍在初始化之后，未声称预加载对 RNG 无扰动。

## CPU 验证

### 14 个新回归与固定负载 77 项

使用未修改公共 server 的 `_load_p0_shadow_api` 与 `_load_p0_action_draft_b2_predictor` 方法 AST，绑定到最小 CPU 对象；真实导入现有 verifier API，真实读取原 `A_off.pt`，不是用 lambda 替代加载逻辑。

- fresh predictor / 默认 off：直接调用原 loader 仍拒绝，原安全保护保持。
- 4 个调用前模式：预加载成功，准确恢复原模式，输出维度 112，eval/CPU，参数逐项等于真实 checkpoint 的 `model_state_dict`。
- cached predictor：同一对象，不再次读取 checkpoint。
- checkpoint 缺失、几何不匹配、API 文件缺失：异常上抛，off 恢复。
- 原真实 predictor 注入 `.to` / `.eval` 异常：调用原 loader 的 CPU map 合同，异常上抛、off 恢复，没有重试。
- 完整任务 Runtime constructor 配 fresh 真实 loader：完成 CPU ready；加载失败时不发布 ready、不安装 save/forward 包装。
- 修复后旧 GPU job 的 CPU 实现 SHA 不再匹配，原执行 guard 拒绝续用；不修改旧 gate 以强行通过。

`libero-va` / PyTorch `2.9.0+cpu`：**77 passed**，其中原 63 项 + 新 14 项。合并旧成本包装 76、旧不扰动 43、最小冻结 target 13：**209 passed in 15.20s**，实际测试进程 exit 0。

### 原 va 环境：直接真实模块探针

`va` / PyTorch `2.9.0+cu126`，`CUDA_VISIBLE_DEVICES=''`、GPU 两个执行开关均未设置。直接导入真实 `wan_va_server`，使用原 `VA_Server` 的绑定方法；通过 `__new__` 仅建立最小 CPU 上下文，**不调用大模型构造函数**。

- 原 fresh/off 保护确实拒绝；修复后的实际 Runtime constructor 与原实际 A_off loader 完成 CPU 预加载。
- 实际 checkpoint 是 `/rjs/cxz/论文探索/投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt`，SHA `d55f59c3b467034d21bc21895136bb797c03a752d6934f854ac5e0ee7e031e7c`。
- 模式恢复 off、112 维、eval、参数全在 CPU、权重逐项一致，cached 对象身份保持；`torch.cuda.is_initialized()` 为 false。
- 没有 target/VAE/text 大模型加载、policy inference、EGL、teacher 或训练。模块导入时“CUDA 不可用，禁用 autocast”的 warning 保存在日志，不是 GPU 推理。

## 历史保留与执行边界

- 修复前 11 个实现文件已经按旧 CPU gate SHA 原样归档；两个修改文件 diff 保存，旧实现可审计。
- 1175 个旧源码/输入/研究证据/历史记录 SHA 与 9 个 target size/mtime 核验通过；旧失败 job 的 33 个文件完全不变，原终态仍是 **FAIL**。
- **本轮 0 新 GPU、0 EGL、0 完整 episode、0 完整 target 固定 chunk；无训练/RL/联训、服务重启、依赖安装。** CPU 合成 infer 是工程测试，不是模型/GPU 固定 chunk，不消耗或续用旧执行额度。
- 新目录是 CPU-only 修复证据，没有 GPU plan、launch 或 supervisor；GPU_run_approved=false。未覆盖旧 gate、plan、日志或失败证据。
- tile 路线只读核查，没有改写或回滚。HW 为投机采样入口追加，并保留其他会话的共享记录。

## 下一步：单独确认新 GPU 关

**尚未启动，不自动推进：**

1. 用户单独确认新 GPU job；建议维持原上限：**0 新完整 episode、最多 2 次 EGL 初态、168 固定完整 chunk、90 分钟 GPU 安全限时**，失败停止、无重试/补调用、暂不训练。
2. 建立新 job，重新冻结修复后实现与计划，重新跑完整 CPU 输入/工程门、确认新鲜 GPU/端口/锁。**本 CPU 修复结果不是可复用的 GPU 执行 gate；旧失败 job 不恢复。**
3. 逐关执行：两初态 → prefix/四 fixture 全缓存恢复 → G1 包装不扰动与真实 KV → S1/L2/L1/S2 成本。任何失败停止并记 HW。
4. 通过并汇总成本后，再讨论成本感知启停或世界条件/真实状态 draft 训练；不因 CPU 修复通过自动开训、改阈值或联训。

## 证据

- [CPU-only 修复目录](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019)
- [CPU 修复验证](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/CPU-REPAIR-VALIDATION.json)
- [209 项实际回归日志](../../../review/OMISSIONS.md)；[测试进程与命令](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/combined-test-process.json)
- [直接真实模块 CPU 探针](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/real-module-CPU-result.json)；[原环境日志](../../../review/OMISSIONS.md)
- [旧实现归档](../../../review/OMISSIONS.md)；[包装 diff](../../../review/OMISSIONS.md)
- [HW 写后与历史保留核验](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/POST-RECORDING-AUDIT.json)；[CPU-only 终态](../../../投机采样/results/runtime/p0-fixed-load-preload-cpu-fix-20261011-195019/STATUS.json)
- [上次失败汇总，历史不改写](FIXED-LOAD-COST-ATTEMPT-20261011-174601.md)
