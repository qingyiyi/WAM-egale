# 原 torch / CFG2 共享 KV：完整 verifier 性能对照

本轮只新增私有测试包装。原 torch 参考、已逐位对齐的 shared_prefix_torch_v2 CUDA 源码与二进制、默认关闭的实例 adapter 均不改；旧失败和完整数值门证据冻结复用。`.pipeline/state.yaml`、rules、cycle 及公共模型/server/config 不改。

## 顺序、预算与验收

1. 新进程加载同一冻结模型、A_off、T-init3-chunk14 请求、threshold/selector/seed，原 cache/torch attention、text CPU/VAE CUDA/FSDP 关闭。原路径 capture 52 次，在 action terminal 前停止，得到同一 29 个变化输入及候选前历史。逐个输出哈希须等于既有原参考。
2. R 单 query 资源探测 1 次；R/S 各完整 29 query 预热，共 58 次，每次仍检查字节、history/metadata/RNG、lease 清理及显存阈值。
3. 六组 `[R,S]、[S,R]` 交替，各臂均完整 29 次，共 348 次。继承冻结 Runtime.sequence 原计时实现：包含输入 staging、lease/setup/keys、写前/返回后持久计数和 release，CUDA event 是整体 span。历史恢复、输出 CPU 拷贝、字节比较与保存不在计时中；所有 OFF 样本无 profiler、无 attribution hook，不能删除样本。
4. 沿用原六组分析：两臂 MAD/median ≤7%；positive pilot 要求 S/R 中位 ≤0.95 且六组均 <1；negative pilot 要求中位 ≥1.05 且六组均 >1；否则 flat/mixed。任何不逐位输出立即失败，不放宽容差。
5. 若不是稳定 positive pilot，在 OFF 全部完成后追加 R/S 首个 query 各 1 次 profile，最多 2 次。仅用于解释缓存构建、拷贝、attention 与其他模块成本，不替代 OFF 数值和正式性能样本。
6. 最后 R/S 各一个 query，共 2 次，复查数值、history/RNG 与释放。worker 退出、端口释放、NVML 无自有进程，计数 journal 独立恢复。

本轮最多 463 次 target 写前意图，0 直接 attention，1 个新 GPU worker / 1 次启动 / 无重试；含加载、IO、同步、清理最多 900 GPU 秒。累计已用 507 次、1256.9490353371948 秒，最坏累计 970 次、2156.9490353371948 秒。

旧累计 target 上限 900，CPU 预检可以准备，但 GPU 必须持有绑定 plan 哈希的明确用户预算回执，将累计 target 上限调至 1000；累计 GPU 时间上限仍为 2700 秒。旧计划/日志保持原 900，不覆盖历史。默认 `--prepare` 只执行 CPU，`--job` 必须经过回执、CPU gate、稳定空闲与 PID 身份 guard。

## 证据与适用范围

保存原 29 个 gold 和 12 份 OFF 的实际 CPU 输出，供独立逐字节复查；输入及历史维持 RAM，不新增多 GiB 训练数据。复用前轮 30 层 QKV/attention、全29输出、12失效和 R/HC/S隔离的正式 PASS，相关源码/二进制均冻结。此次预热和每个样本再次验证所有实际输出及状态。

0 新 episode、0 EGL、0 完整 policy chunk、0 环境动作、0 训练。只能回答单 worker、固定 30 层/29query 的 verifier 成本；性能 positive 后才规划任务闭环。不将“测量稳定”解释成“有收益”，不将共享存储解释成七候选 CTA 片上复用。

执行及脚本审查为同一主代理，`execution.mode=self`；独立 CPU 审计脚本验证原始 bytes、计数与 SHA，不宣称独立代理复核。
