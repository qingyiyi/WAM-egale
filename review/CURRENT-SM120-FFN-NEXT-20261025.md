# 当前SM120：保留S，FFN1 mbarrier流水W未采用

最新[W结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/REVIEW_RETURN.md)、[W完整18阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/STAGE-TABLE.md)、[固定源码/结果入口](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/SOURCE-AND-RESULT-ENTRY.json)。

W只改FFN1：1T/4M/3E，448个M64N128输出tile；48个K64完整累加后bias/BF16/GELU，热循环mbarrier，E线程写global ACT，保留FFN阶段同步。不是TMA输出写回，也不是整层DAG。FFN2七段数值代码不改，ACT覆盖计数按新tile schema调整。

三KV/两进程的27tensor、七段partial/running、有效KV/输入/RNG、角色计数通过；完整层比同场S慢0.85%/0.79%/0.83%，未采用，恢复S。W和S同场原Torch加速分别约1.274/1.785/2.226与1.284/1.799/2.244。所有计时双方eager/noGraph/warm CrossKV，仅block replay，非机器人E2E。tile与分工同时改变，不能单独归因mbarrier或TMA；无硬件计数器因果结论。

实际采用仍为[FFN两阶段版S](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)，[S历史18阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/STAGE-TABLE.md)。原A与DSM负结果保留。W只是独立冻结证据，不要从此目录的whole_layer.cu误认为默认入口已换为W。

W本轮2865调用、14.79分钟生命周期、约137.47MiB产物，关键文件回传按字节核验、owned清理完成，public S再验通过。四项科学权限均false。后续先定位同tile输入等待与GELU写回成本；TMA store仍是讨论提案，未实施。
