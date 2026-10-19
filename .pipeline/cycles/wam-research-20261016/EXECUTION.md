---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-19T21:08:07.097619+08:00'
---
# 最近执行检查点

## 2026-10-19T21:08:07.097619+08:00：M32单层persistent

[M32结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)、[两进程九点计时](../../experiments/persistent-tile-remote-20261019/PERFORMANCE.md)。用户最新限定一层完整Transformer block，已取消本轮30层迁移/执行；最终模型E2E后续本机。远端~/cxz隔离CUDA12.4/Torch2.9 cu126编译；baseline与依赖88项不变。

attention/FFN1阶段内实现常驻CTA固定步长tile循环；完整self/cross/FFN与norm/RoPE/调制/残差仍在同一层。QKV/Wo/FFN2仍cuBLASLt，尚无跨阶段ready、warp专职分工或新shared-prefix复用。默认sequential，静态/persistent及FFN1独立开关。本机旧M31 DSO需重编新ABI，远端新DSO已验证。

9真实block（B2/B14 action及B2 Q128 video×三KV）输出逐位；90有效KV连续状态、输入/RNG/参数/API门通过。tile/persistent相对独立顺序路径24中间量，action融合23；48原语检查确认每tile恰好一次与常驻CTA实际多tile执行。无baseline模块/旧DSO加载，原Torch为远端oracle。

两进程378个原始计时样本：persistent+FFN1相对静态+FFN1，B14慢4.42%–6.26%，B2 action慢2.56%–2.80%；未取得新增persistent收益。A/A最大10.07%保留，GPU1/host共享负载及三样本中位限制明确；不是最强B14/Graph或E2E。

五GPU作业77.927秒/731block/48原语/0full30/0episode，全部owned退出。首计时在0block因NVML宿主/容器PID误判退出，修正资源工具后新attempt2/3；传输失败、原收据均保留。CUDA active时间UNKNOWN。原3h墙钟/30min GPU预算收口，不转授下一轮。

Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

最新用户修正优先于旧full30计划，已经写入约束记忆。未改保护state/rules或投机源码；无模型大文件hash/系统升级。所有原始证据见M32结果链接。

前完整执行见[归档](history/M32-before-summary-20261019/EXECUTION.md)。
