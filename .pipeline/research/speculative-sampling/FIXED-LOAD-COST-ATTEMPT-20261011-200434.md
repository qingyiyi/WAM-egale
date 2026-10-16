# 新固定负载 GPU job：启动失败停止与 CPU 生命周期定位

日期：2026-10-11（Asia/Shanghai）  
归属：投机采样 / action-only / 冻结 target；**不是 tile 算子研究**。  
结论：**GPU job FAIL，已停止并清理；CPU 原因定位 PASS，未修复实现、未重新启动。恢复、不扰动和成本结果仍为 NOT_RUN。**

## 1. 本次授权与实际执行

用户确认新建 job，先 CPU 预检，再按恢复→不扰动→成本对照推进；失败停止、不补跑、暂不训练。

- 新 job：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-fixed-load-cost-20261011-200434`。
- 上限：0 完整 episode、最多 2 次 EGL 初始化、168 次完整固定 chunk 调用、90 分钟 GPU 安全限时。
- 原 target、实际 `A_off.pt`、global selector、阈值、K2B2/no-bonus、原 cache 和原 denoising 工作定义不变。
- 未恢复旧失败 job，未复用旧 CPU gate；本次重新冻结 plan/11 个实现文件，执行新的 CPU 与资源门。

## 2. 关卡结果与预算

| 关卡 | 实际结果 | 证据 / 边界 |
|---|---|---|
| CPU 回归 | PASS：209 项，16.38 秒 | 实际 pytest 退出码 0；包括 77 项固定负载回归。不是整套真实 WAM 新启动生命周期验证 |
| 原 va 环境真实接口 CPU 探针 | PASS，但覆盖有限 | 真实 server loader 和实际 A_off 参数加载；探针手动设置了 `s.action_per_frame=4`，遗漏 reset 前状态 |
| 冻结 / 输入工程门 | PASS | 1216 个源码/输入/旧证据 SHA、11 个实现 SHA、plan SHA；9 个 target 文件 size/mtime；84 个旧 artifact、440 个原双摄窗口帧 |
| GPU 资源门 | PASS | 启动前显存 0 MiB、无 compute-apps；磁盘、RAM、端口满足要求 |
| 两模型启动 | **FAIL** | 两个 worker 均在 `Runtime.__init__ → preload_action_draft` 报相同缺字段错误 |
| EGL 初态 / fixture 恢复 | NOT_RUN | 尚未启动 EGL 初态 worker，无真实 GPU fixture |
| 不扰动 G1 | NOT_RUN | 没有可比较输出 |
| 短/长 KV × 单/双需求成本 | NOT_RUN | 没有 action/policy 延迟、需求负载或成本差异结论 |
| 清理 | PASS（本 job ownership） | owned process groups 已退出，所属端口无监听，任务 tmux 已退出；不等于 GPU 之后一直空闲 |

GPU 计时约从 **20:07:05** 开始。失败快照耗时 **50.12 秒**，清理后状态记录约 **50.90 秒**，未触发 90 分钟超时。

实际消耗：**0 完整 episode、0/2 EGL 初始化、0/168 完整固定 chunk、0 prefix replay**。两个模型进程启动不计作两次 EGL 初始化。

**失败后没有重试、补跑、修改后重启或训练。剩余预算不是再次执行的许可。**

后续 CPU 整理期间的 **20:16:22** 资源快照发现 GPU 已被非本 job 进程 `3421190` 使用（当时约 19251 MiB）；只读 `ps` 核实用户是 **hlj**、进程组 `3421189`、启动时间 **20:15:43**，晚于本任务停止。它不属于本任务的两组 owned processes `3396431/3396432`，没有被信号、终止或修改。因此 `POST-GPU-RESOURCE-CHECK.json` 的 FAIL 表示**当时 GPU 不空闲**，不能写成“本任务清理失败”或“当前显存归零”。未来 GPU 除重新授权外，也需通过新的独占/空闲资源门；不干预他人任务。

## 3. 根因：真实 server reset 前后生命周期不匹配

真实 worker 错误：

```text
AttributeError: 'VA_Server' object has no attribute 'action_per_frame'
```

证据链：

1. 原公共 server 的 `_load_p0_action_draft_b2_predictor()` 在 `wan_va_server.py:839` 计算 `len(channels) * self.action_per_frame * self.job_config.frame_chunk_size`。
2. 原 `VA_Server.__init__()` **不设置** `self.action_per_frame`。该字段由原 `_reset()` 在同文件 **2121 行**从 `job_config.action_per_frame` 建立，并与其余几何、cache、mask、norm 等 reset 状态一起初始化。
3. 本任务 `worker.py:47` 在 `Runtime.__init__()` 中预加载。原 reset 请求直到 `Runtime.build()` 的 **133 行**才发生；因此新启动 server 必然缺字段。
4. 前次修复的临时 `verifier_k2_b2` / `try/finally` 已绕过原 fresh/off guard，并在此次错误后恢复 off；但只是修好了模式保护边界，**没有修好 reset 生命周期顺序**。
5. 本次 CPU 探针复制了前次探针，在 **34 行**人工设置 `s.action_per_frame=4`。CPU 回归上下文同样预填了该字段，使“真实 loader + 真实 checkpoint”的通过不能检验“真实新启动、尚未 reset”的状态。**这是任务包装和 CPU 覆盖遗漏，不是 target、A_off 质量或投机算法失败。**

CPU-only 后验诊断已通过：

- 隐藏 CUDA，关闭 GPU 执行标志，导入未改动原 server 模块和冻结 Runtime。
- 不手工设置 `self.action_per_frame`，直接进入当前 Runtime，复现同样 AttributeError；off 模式恢复，forward/save hook 未安装，没有生成 ready。
- 执行原 `_reset(prompt=None)` 后，再调用当前预加载 helper：真实 A_off 输出维度 112，eval、CPU 权重逐项与 checkpoint 相等。
- 此反向检查中的 transformer/VAE cache 操作是明确的 CPU stub，未构造真实 target、未分配真实 GPU cache、未编码真实 prompt、未执行 prefix/infer。**只支持生命周期根因和修复方向，不是新预检 gate 或 GPU 通过。**

## 4. 本轮保存与未变更范围

诊断后再次检查：1216 个冻结文件 SHA、11 个实现 SHA、plan SHA、实际 A_off SHA 均不变；9 个 target 文件 size/mtime 不变。target 检查是元数据，不冒充全量权重哈希验证。

本轮没有修改 public server/client、包装实现、测试实现、target/draft 权重、阈值、bonus、cache/scheduler/stride，也未写 tile 文件或受保护的 `.pipeline/state.yaml`、`cycle.yaml`、`rules.yaml`。

记录后的只读边界复查（20:23）中，1216 个投机采样冻结文件、11 个实现和 9 个 target 元数据仍一致；另发现 4 个 **tile 路线文档**相对本轮开始时快照发生变化。本任务没有写它们，不覆盖、不回滚；只保留边界观察，不能宣称整个共享 workspace 完全未变化，更不将其成果并入本投机采样实验。见 [只读边界观察](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/POST-RECORDING-BOUNDARY-OBSERVATION.json)。

AUTH、CPU、START、FAILED 各阶段均已独立写入 HW；另补本报告、POST-CPU Knowledge record，并增量更新投机采样 RESEARCH、相关索引/compact、PROGRESS 和英文生命周期日志。历史首次失败和前次 CPU 修复报告保留，不能覆盖成成功。冻结 `plan.json` 的 CPU 启动前状态是历史快照；实际终态以 `FAILED.json`、`status.json` 为准。

## 5. 下一步：先确认 CPU 生命周期修复，不重构、不直接再试 GPU

建议下一次只做 CPU 工作，需用户确认后执行：

1. 把任务预加载移动到**原 reset 成功建立推理状态之后**、prefix/fixture seed 与计时之前；constructor 的 ready 只表达 target/server bootstrap 已就绪，不冒称 draft 已就绪。保留默认关闭、原 off guard、异常恢复和失败停止。
2. 补充“新启动→原 reset→draft preload→prefix 前置条件”的实际方法回归；不人工预填 reset 拥有的字段，不以 cached predictor / lambda loader 掩盖首次加载。系统核对其他调用阶段、字段依赖和就绪信号，避免每次只补一个缺字段。
3. 验证 off / 其他原 mode 恢复、错误不安装或泄漏 hook、不写假 ready/fixture、真实 A_off 几何/参数/设备与 cache/RNG 顺序约束；明确哪些重组件仍是 CPU stub，不能宣称完整真实 WAM 启动已测。
4. 通过后冻结新实现、重新跑完整 CPU gate 并记录 HW；**再单独确认新的 GPU job/预算，并重新确认 GPU 空闲**。两个已失败的 one-shot job 均不恢复，不覆盖旧 gate。

**研究路线不重构，仍先做成本对照；但必须先补齐工程生命周期。没有成本对照结果，不能判断短/长 KV 或单/双需求下 T/A_off 谁更快，也不能推进新训练、RL 或大小模型联训。**

## 6. 原始证据

- [本次 CPU gate](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/preflight/cpu-gate.json)
- [209 项实际日志](../../../review/OMISSIONS.md)
- [探针及其预填字段](../../../review/OMISSIONS.md)
- [失败终态](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/FAILED.json)；[预算状态](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/status.json)
- [worker0](../../../review/OMISSIONS.md)；[worker1](../../../review/OMISSIONS.md)
- [CPU 生命周期诊断](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/CPU-LIFECYCLE-DIAGNOSIS.json)；[实际进程退出码](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/CPU-lifecycle-diagnosis-process.json)
- [所属进程清理](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/OWNERSHIP-CLEANUP.json)；[后验 GPU/端口/tmux 检查](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/POST-GPU-RESOURCE-CHECK.json)；[非本任务占用说明](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-200434/POST-GPU-OWNERSHIP-INTERPRETATION.json)
- [研究主线](RESEARCH.md)
