# 当前执行：阶段同步版的按需调制融合

当前单项优化已完成：将temb+固定调制表的FP32加法移到两处输入调制和两处门控残差，普通路径不物化MOD。保留阶段同步，不恢复ready队列。

同轮远端A100、两独立进程五臂Graph，短/中/长旧版2.292/2.459/2.472ms→新版2.269/2.435/2.449ms，下降1.005%/0.979%/0.942%。相对最强同工作量guarded仍慢9.59%/11.34%/11.27%。不是完整模型/机器人E2E。

9真实签名和21结构有限逐位/状态通过。普通/debug一致；debug额外生成MOD以审计，计时用普通路径。普通路径少2016tile、一次grid同步、15.75MiB中间张量；不是实测HBM节省或峰值显存结论。

两机当前source9edf85a2、DSO41af9a65，旧11和ready负结果保留。GPU已清理，累计7326.682/36000秒；四科学权限false，旧paused72身份不改。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)、[当前完整阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[HW](../.pipeline/PROGRESS.compact.md)。下一项先讨论Norm1与输入调制，不自动新跑。
