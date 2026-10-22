# WAM当前入口

先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。动态结果与预算以Cycle及原证据为准。

M39独立8warp Self QKV已采用，当前完整阶段表全部重测；原Torch完整单层1.2661/1.8074/2.9322×。首版混合函数负结果保留；其余三小投影仍有余差。GPU0已清理、远端无硬件计数器；GPU1固定RoboTwin队列19/1800自动接续，四科学权限false。

[本轮结果](experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESULT.md)、[当前完整阶段表](experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/STAGE-TABLE.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
