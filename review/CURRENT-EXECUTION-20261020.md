# 当前：简易单层与逐阶段对齐

固定本文件所在commit审查；paused72、旧RESULTS.json/FAILED/retention及四质量权限false不改。

简易版已通过9真实签名/23中间量/18有效KV转移等有限逐位门。两进程同输入交错计时无稳定退化，中位数比复杂版低0.4%–0.7%。复杂活动入口已删除，历史源码/失败归档。当前仅维护`算子研究/wam_ops/optimized/whole_layer.py`与`csrc/whole_layer.cu`主入口。

完整层仍未对齐native：短约1.85–1.93倍、中1.86倍、长1.71倍。uncached/eager、输入准备排除、GPU0无其他任务；非强cached/Graph或E2E。阶段仍grid.sync。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)、[全部阶段矩阵](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/STAGE-RESULT.md)：cross K/V独立仅慢9%–10%；FFN1约1.84倍；cross-K RMS约3.4倍、GELU约6倍、table约5倍、layout约21–36倍。独立/嵌入资源和缓存历史不同，不能仅凭C/A归因global sync；NCU未知。

下一步先简易版pointwise与B14只读layout，再cross-K RMS、FFN1/cross-attention，最后context K/V ready填尾波。没有新增这几项优化收益。当前GPU0清理完毕；本阶段662/720block、4524/6000stage，106.745秒GPU worker生命周期，全部失败计入。[资源](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/RESOURCE-AND-STOP.md)。

RoboTwin GPU1固定42Target采集继续：快照新增19/42、15成功，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。目标Target48 FIT36/CAL12，非TF评测；不自动训练，截止次日03:37:02。用户算子10h授权没有改变RoboTwin预算/截止。

请网页端评审直接阅读当前简易源码及A/B/C矩阵，优先提出SM80可执行、保持原归约/BF16和合法KV语义的改动；不要恢复旧复杂版、假设跨CTA shared memory，或声称已经完全等速。
