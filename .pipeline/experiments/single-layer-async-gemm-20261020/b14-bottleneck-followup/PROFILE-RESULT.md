# 当前B14完整单层瓶颈复核

OBSERVED：三份真实B14输入，短/中/长KV分别288/1152/2160；当前已对齐版本的插桩复本，12次完整block、150次同输入原Torch Linear调用。全部配对中间量/输出逐位、有效KV通过；30组原语输出逐位。guard COMPLETE，自有进程回收。

| KV | 插桩整层ms | GEMM合计ms/占比 | cross K/V ms | FFN1/2 ms | self+cross attention ms |
|---|---:|---:|---:|---:|---:|
| 288 | 11.78624 | 8.967168 / 76.08% | 6.373376 | 1.433600 | 0.507904 |
| 1152 | 12.13776 | 8.937792 / 73.64% | 6.360064 | 1.425056 | 0.888416 |
| 2160 | 10.058816 | 7.111840 / 70.70% | 5.037152 | 1.157088 | 1.095072 |

同输入Torch Linear事件计时：cross K/V每次约0.59–0.74ms，自有阶段约2.50–3.21ms；FFN1约0.198–0.232ms，自有0.643–0.786ms；FFN2约0.141–0.159ms，自有0.514–0.647ms。阶段包括领取/等待/边界同步，Torch事件含其launch边界，因此不是纯MMA指令吞吐倍数。原Chrome trace以cat=kernel筛出的设备事件单列在*-kernels.json；原RESULT中primitive_kernel_events还包括annotation/memset，不将其直接求和称kernel时间。

这次比较的GEMM包括self Q/K/V/out、cross Q/K/V/out和FFN1/2，不只FFN。B14 action线性层M224，cross条件M7168；隐藏D3072、FFN14336。attention仍14个Q16逻辑实例，不是Q224序列。

HYPOTHESIS：成熟SM80设备GEMM的多级异步装载、shared-memory布局和warp MMA，可能显著改善目前主瓶颈。这里只建立优化优先级；没有证明寄存器溢出、HBM或某个具体stall是根因。

UNKNOWN：无NCU achieved occupancy/DRAM/L2/warp stall新证据。插桩版255 registers、121376B shared、1704B localSizeBytes，后者不是溢出流量实测。插桩改变二进制，不能把本表作为最终普通版速度，也不是强cached/Graph、verifier或机器人E2E。

RECOMMENDATION：先在原single-kernel边界内替换大M GEMM mainloop，固定调度/分段/bias顺序；数值/状态门通过才测速度。cross K/V缓存是强baseline已有工程，不把跳过这部分重复计算称新MegaTransformer贡献。

证据：[原始RESULT](RESULT.json)、[摘要](SUMMARY.json)、[资源终态](STATUS.json)、[源清单](SOURCE-MANIFEST.json)、[预算](BUDGET.md)。旧M38与当前profile不能混用。
