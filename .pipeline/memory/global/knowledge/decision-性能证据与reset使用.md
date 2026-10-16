---
kind: decision
name: 性能证据与reset使用
level: reference
scope: research
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
- .pipeline/research/speculative-sampling/CURRENT-CHUNK-LENGTHS-RESULT-20261014.md
- .pipeline/research/speculative-sampling/SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md
---

# 性能证据与 reset 使用

旧A_off完整policy chunk短/中/长KV288/1152/2160正式加速1.601926×/1.544039×/1.500061×，45完整chunk、18/18配对胜出。同算法RAW/O3输出/路径/RNG/30层有效KV核验，不代表近似投机与T动作逐位相同。

旧4对任务两方4/4成功，episode合计1033.403/898.071=1.150692×，不含环境构造/模型加载；冷server到client退出1.090947×。旧账本reset46.81%、policy44.20%、环境动作4.16%。不能称新TF或MegaTransformer收益。

固定真实30层29-query共享前缀pilot 5.170557→3.359750秒，额外allocated peak891.035→257.085MiB；348输出及history/RNG核验。父清理guard失败原样保留，性能后有独立CPU核对；不是完整episode结果。

最终text bank保留原512-token BF16，11条约44MiB；CPU7首次构建约706秒。裁padding0/11逐位失败，未用。8完整任务/4对验证动作、KV、VAE、RNG等价；reset内部127.8–154.2秒→24.8–27.5毫秒，RPC约.1秒。

未来采样/质量/性能可以在等价验证后统一启用优化reset，双方baseline同样处理，首次构建/模型启动另报；不能推算并宣称新E2E已测。新TF耗时和target调用数仅诊断，正式45chunk未通过准入、未跑。
