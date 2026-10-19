# 单层、单kernel、warp分工：实验入口

入口：[whole_layer.py](whole_layer.py) → [csrc/whole_layer.cu](csrc/whole_layer.cu)，独立构建[build_whole_layer.py](build_whole_layer.py)。**研究实现，未通过原逐位门；不接入模型默认路径。** baseline、M32 DSO及投机源码不改。

本版将一层全部计算放在一次`cudaLaunchCooperativeKernel`内：调制表、三次LayerNorm、四次Q/K RMSNorm、self/cross QKV与输出投影、RoPE、两种attention、FFN两层和GELU、残差、KV metadata/私有写入与恢复。host仅检查接口、绑定参数和分配workspace。不调用host cuBLASLt，不在设备里再启动子kernel。

## 线程与任务

一个常驻CTA有8个warp：

| warp | 职责 |
|---|---|
| 0 | 从当前阶段队列领取tile；全层cache metadata由首CTA首线程处理 |
| 1 | GEMM activation或attention query加载 |
| 2 | GEMM weight或attention K/V加载 |
| 3–6 | BF16 Tensor Core MMA、softmax、Welford归约和pointwise |
| 7 | GEMM bias/BF16舍入、attention在线输出合并、写回 |

GEMM每CTA处理64×64输出tile，K按32推进；4个计算warp分别负责32×32。attention每CTA处理一个candidate/head中的16个query，K按64推进。B14仍是14个Q16实例，各候选private KV可见性独立，不把7候选拼为可互相注意的Q112。

同一阶段的tile可由任意常驻CTA领取。阶段边界用cooperative grid barrier保证完整归约/依赖ready；norm的计算warp组使用显式128线程named barrier，不让旧CTA barrier只被部分warp调用。CTA数由实际寄存器/共享内存的occupancy上限确定，禁止超额cooperative驻留导致死锁。

这是**整层范围的阶段DAG执行器**，不是细粒度跨阶段流水。当前中间量仍写入global workspace；没有宣称权重/KV只从HBM读一次，没有SM90动态寄存器分配，也没有多个候选共用private KV。粗屏障和高资源常驻可能更慢，待实际profile判断。

## 数值与接口

已登记真实输入为B2/B14 action Q16、B2 video Q128，D3072/H24/head128/FFN14336，三种历史长度。输入/输出BF16；调制和LayerNorm保留FP32，RoPE按原FP64算术，GEMM与GELU之间保留BF16边界。

新WMMA归约及在线attention与原cuBLASLt/Flash的实现顺序不同。代数一致不能推定逐位一致，也不能继承M32精度或闭环证书。原逐位/状态门不降低；若失败，只能报告实际误差并保留未准入研究实现。单kernel结构、tile恰好一次、有效KV/RNG/输出与速度分别检查。

验证与资源证据：[M34登记](../../../.pipeline/experiments/single-layer-warp-specialized-20261019/PLAN.md)。CPU已编译；本机9真实输入完成，结构成立但9输出原逐位门失败，不集成。详情见[M34本机结果](../../../.pipeline/experiments/single-layer-warp-specialized-20261019/LOCAL-CORRECTNESS.md)，远端9 trace也确认单kernel、两独立进程计时相对native慢5.12–11.28倍；[完整数据](../../../.pipeline/experiments/single-layer-warp-specialized-20261019/RESULT.md)已收口。进一步有界诊断不改主源码/DSO。只有profiler实际观察一次整层kernel才计结构门通过；源码中单launch语句不是运行证据。
