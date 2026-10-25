# 当前SM120算子入口

更新2026-10-25：采用S不变。FFN1成本诊断4296调用完成，主循环加载/计算/内部等待约占主体tile95.6%；尚未精确区分硬件stall。MegaMoE源码差距已核验。不是新的优化收益或闭环放行。

- [结果/下一步](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-GEMM-GAP-20261025/REVIEW_RETURN.md)
- [完整18阶段](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-GEMM-GAP-20261025/STAGE-TABLE.md)
- [MegaMoE机制差距](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-GEMM-GAP-20261025/SOURCE-REVIEW.md)
- [短状态](../.pipeline/PROGRESS.compact.md)

先前A2/共享A/B/W及C2/C4负结果全部保留。四项科学权限仍false。5090阈值DEV仅1组完整四臂，未定最终tau；已归档临时资产。公开镜像不包含权重、原tensor、DSO和完整CUPTI trace；源与派生JSON路径保留，不能据公开包声称已重放原输入。
