# Prompt / cross-KV：保留结构检查的模块枚举优化，真实 GPU 三档三组完整对照

用户授权：“你确认cpu没问题以后，直接把gpu的也做到没问题……中途你自己决定就行，我只要结果”；随后将累计 GPU 时间扩展到 5 小时。沿用 HW chat 研究记录，原型私有默认关闭，不改生产代码或受保护状态。

## 已完成的 CPU gate

私有 CPU 原型 28 项全部通过：FP32/BF16、原生产 loader、30 层/27 动态查询及真实完整文本、S0 组合逐位一致。独立审计重新读取 84 个输出对/564480 元素，并通过真实 ATen 分派确认文本矩阵运算 54→2、cross KV 1620→60、Q 810→810。第一次矩阵形状相同。首次失败的 None cross-cache 状态处理已修复，原失败日志保留。

本版本额外通过 31 项 CPU 检查，新增子模块替换、参数对象替换、注册新缓冲三类失效检查；独立审计重新读取保存张量并验证真实 ATen 运算次数。模块列表仅在租约入口枚举一次，每次前后检查仍验证全部子模块/参数/缓冲字典的身份、成员、对象及 tensor 版本；首次矩阵运算和缓存内容不变。详见 WATCHED-GRAPH.md。此前完整扫描版短/中档中位耗时相对 S0 增加 7.56%/7.45%，长档虽中位降低 13.39%，仅 4/6 胜出；全部结果和超时失败均保留，不合并跨 job 计时。

新 GPU wrapper 必须另外完成 CPU 预检，使用同一 Runtime 时钟和持久 counter，对规范 prompt 绑定、三组切换、S0/P 输出、异常释放、准入、预算和统计做实际接口检查。

## 对照定义

R：原 Torch 数学；S0：冻结的原 Torch 逐位对齐共享 self-attention 前缀；P：S0 加文本投影与每层 cross K/V 的完整 14 行缓存。新增行证明优化不启用。Q、norm_k、SDPA、FFN、当前候选、CFG 顺序和首次投影形状均保留。GPU adapter 来自已过 gate 的 CPU 代码，仅增加显式、plan 绑定的 CUDA 边界以及 GPU autocast/TF32 上下文保护。

真实 short KV288 / middle KV1152 / long KV2160，预期 27/28/29 自然 verifier 查询；原 snapshot 或记录前缀由原 server 恢复，不裁剪 KV。捕获停在动作 terminal 之前；0完整 policy chunk、0EGL、0episode、0训练。

捕获后逐个验证所有 CPU text_emb 的实际字节相等，再在每个 P 序列首 query 的 timer 内建立一次 CUDA 规范 prompt，并由该租约绑定跨查询复用。P 会省掉重复文本 staging；报告需注明该效果包含在新增优化中。CPU 输入证据检查不在 verifier timer，三组使用同一已捕获 RAM 数据。保留真实 14 行首次 GEMM，避免 14→2 新矩阵形状误差。

每组完整预热、六轮全部排列平衡 OFF 计时。每个输出同轮原 Torch 必须逐位一致；P 预热对 61 个实际 GPU 投影结果逐字节计算 SHA-256，与同一输入原 Torch 的实测 GPU 结果哈希核对；保存哈希，不保存全部 2.5 GiB 中间张量。全部 timed 输出保存独立 CPU 审计，不删样。中位至少降低 5%、六轮均胜出且 MAD/median≤7%才称正向。失稳/混合/负向均如实保留。

首次 setup、CUDA 规范 prompt staging、缓存 clone、全部 keys、fsync intent journal、27/28/29 调用和释放均计入每组 timer。输出字节检查与 all30 history/RNG 检查位于 timer 外。记录峰值 allocated/extra、payload；预计 2.502 GiB projection cache，上限 3 GiB，extra peak≤12 GiB、free≥8 GiB。

## 有限资源及授权

累计前向上限由 agent 在用户自主决策授权内设为 7500（已有 4667 不退还），累计 GPU 时间上限按明确新授权为 18000 秒（已有 3729.3681666012853 秒不退还）。本 job 仅 1 个 owned GPU worker、1 次启动、最多 1800 秒，stage ledger 明确有界，无本 job 重试。CPU gate 后写入绑定 plan 哈希的授权回执，包含两条原始授权文字。若本次失败，保留全部计数及失败证据，任何新 job 必须重新累计而非覆盖。

GPU 启动前须 15 次稳定 idle、A100 40 GiB、磁盘≥20 GiB、CPU≥10 GiB、端口空闲。持续所有权检查，仅能清理明确属于本 job 的 PID/process group。模型返回计数不冒充设备完成，最终同步输出、journal CLOSE、exit 和资源清理均须通过。
