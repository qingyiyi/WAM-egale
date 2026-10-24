# WAM当前入口

存储整理进行中：临时目录统一/rjs/cxz/tmp；623项闲置/tmp已归档清除，公共模型组件去重已释放13.21GiB。约120.87GiB大文件冷归档到BAAI，当前按原块统计已释放32.166GiB；按文件/闭合小批校验即释放，每30秒跟踪，不等整批；[状态与恢复](maintenance/storage-placement-20261024/PLAN.md)。用户新分工：A100质量、5090性能、本机轻量开发。

SM120 FFN新一轮已收口：[结果](experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)、[全阶段表](experiments/sm120-ffn-next-20261024/STAGE-TABLE.md)。PAIR2048x2七段有限逐位通过但整层慢6.25%–7.89%，NO_GO，保留SELF-QKV-TMA-4MMA；当前原Torch/A完整层约1.251/1.758–1.761/2.191–2.201×。5090 owned清理，双A100接续RoboTwin，01:02为405/600完整组三臂；新OOM中断T未自动重试，见[接续](experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)。

先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)、[当前Cycle](cycles/wam-research-20261016/PLAN.md)。

[上轮FFN负结果](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/REVIEW_RETURN.md)、[上轮阶段表](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/STAGE-TABLE.md)、[固定源码入口](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/SOURCE-AND-RESULT-ENTRY.json)、[旧Self QKV采用](experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)、[双卡接续](experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)。旧失败、预算收据及四false保留。
