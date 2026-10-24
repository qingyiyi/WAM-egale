# WAM当前入口

先读[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)、[短状态](PROGRESS.compact.md)，再读[当前Cycle](cycles/wam-research-20261016/PLAN.md)。v15语义入口；原保护文件/旧记录/原tile保留，不全量sync。

- [5090五环境投机结果与清理](experiments/rtx5090-spec-speed-20261025/RESULT.md)：44组三臂132主表局，当前TF/PF2未获全面质量放行；所有本轮临时权重/环境核验回传后删除，原operator保留。
- [双A100 RoboTwin持续接续](experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)：原600组/160GPUh，旧缺口不重试冒充完成。
- [当前SM120单层结果/阶段表](experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)：PAIR2048x2负结果，保留SelfQKV四MMA A，未集成机器人路径。
- [本机冷归档与恢复](maintenance/storage-placement-20261024/RESULT.md)：25564文件核验后释放120.619GiB，代码/HW/轻量证据保留；默认/rjs/cxz/tmp。

全部原始证据按需读取；派生摘要不能覆盖原失败或把四项false改为true。
