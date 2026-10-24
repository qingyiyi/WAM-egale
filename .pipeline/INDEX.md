# WAM当前入口

Self QKV已迁移一TMA／三epilogue／四MMA并研究采用；原Torch整层短中长1.2475/1.7553/2.1872×，比旧A延迟降2.94%/2.81%/2.25%。三KV有限逐位及公开两流通过。用户追加现有5090持续使用，旧账本保留。双A100继续RoboTwin，13:55为298/600有效完整组，两worker运行。

先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)、[当前Cycle](cycles/wam-research-20261016/PLAN.md)。

[本轮结果](experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)、[完整阶段表](experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/STAGE-TABLE.md)、[固定源码入口](experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/SELECTED.json)、[双卡接续](experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)。旧失败、预算收据及四false保留。
