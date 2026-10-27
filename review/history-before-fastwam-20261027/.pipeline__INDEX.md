# WAM 项目入口

先读[读取策略](CONTEXT-POLICY.md)和[记忆入口](memory/INDEX.md)，读取现行约束后进入[WAM Cycle](cycles/wam-research-20261016/PLAN.md)。[短状态](PROGRESS.compact.md)只保留当前结果、运行作业、阻塞和下一步。原协议/原始证据优先于派生摘要；v15固定入口，不全量sync旧v12。

| 当前主题 | 入口 |
|---|---|
| 本机 Goal 门控验证 | [resume05计划](experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/PLAN.md)；状态按该目录实时读取 |
| 双 A100 RoboTwin | [原固定队列](experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md) |
| 5090 单层算子 | [当前FFN两阶段版S/结果](experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)、[18阶段表](experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/STAGE-TABLE.md)；[FFN1流水W负结果](experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/REVIEW_RETURN.md)，[FFN1双CTA共享B负结果](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/REVIEW_RETURN.md)；[追加共享A结果](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/REVIEW_RETURN.md)；[双K64交接A2负结果](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/REVIEW_RETURN.md)，S仍保留；[最新FFN1慢因/MegaMoE差距](experiments/sm120-ffn-next-20261024/FFN1-GEMM-GAP-20261025/REVIEW_RETURN.md) |
| 五环境投机结果 | [M41结果](experiments/rtx5090-spec-speed-20261025/RESULT.md) |
| 中文逐段写作 | [协作Prompt](prompts/WAM-PAPER-COLLABORATION.md) |

其他 Cycle、历史知识、旧长进度只在对应任务下读取；本入口不授权重启实验；GitHub镜像是发布时快照，不是实时作业状态。[本次精简与完整备份](maintenance/hw-compact-20261025/PLAN.md)。
