# WAM当前入口

先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[当前状态](PROGRESS.compact.md)、[Cycle](cycles/wam-research-20261016/PLAN.md)。固定v15，不全量sync。动态结果与预算以Cycle及原证据为准。

M39外层FFN内联＋尾K64已采用，完整阶段表已完成；原Torch完整单层1.2426/1.7780/2.8859×。核内阶段诊断显示小投影仍有明显余差，FFN区间接近Torch；不外推E2E。GPU0本轮已清理，远端不使用硬件计数器；GPU1固定RoboTwin1800局自动接续，四科学权限false。

[本轮结果](experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESULT.md)、[当前完整阶段表](experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/STAGE-TABLE.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
