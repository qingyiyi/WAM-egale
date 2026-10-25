# 当前SM120：共享A优于共享B，但仍保留S

最新[共享A结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/REVIEW_RETURN.md)、[完整18阶段](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/STAGE-TABLE.md)、[等待定位](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/WAIT-ANALYSIS.md)、[源码/数据入口](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/SOURCE-AND-RESULT-ENTRY.json)。此前[共享B完整负结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/REVIEW_RETURN.md)保留。

S是实际采用的FFN两阶段版；W是448个M64N128、1T4M3E的流水对照。共享B把不同M/同N的CTA配对，共享16KiB权重块；共享A把同M/不同N配对，共享8KiB输入。各CTA仍有自己的输出累加器，K3072/BF16/GELU/FFN2七段和阶段同步不变，原ldmatrix从local shared读取。

两方案三KV/两进程27tensor、七段partial/running、KV/输入/RNG/实际计数通过。共享A与B在追加轮次同场比较：A比B延迟减少21.24%/20.15%/18.76%，但比S增加26.48%/24.20%/21.70%。完整层原Torch/A为1.017/1.445/1.836，原Torch/B为0.801/1.154/1.492，原Torch/S为1.286/1.794/2.235。双方eager/noGraph/warmCrossKV，非机器人E2E。

共享A FFN1约323μs，同场W约152μs；共享B前一轮阶段约560μs（明确历史）。共享A的源EMPTY/接收READY/MMA输入等待约1.76/1.03/1.35–1.37μs每K64；B前一轮约3.32/2.79/2.9μs。插桩与正常计时分开，不把重叠等待相加或称纯DSM带宽。不声称逻辑global请求减少等于HBM减少。

两方案都未采用，保持S；不自动继续第三种设计。A5400次block/10.96分钟、B4836次/17.09分钟；公开S再验、关键文件回传、owned退出核验闭合。四项科学权限false，旧FAILED/retention/训练DEV暴露不改。

最新双A100只读快照19:34：586/600组、1758/1800局，两卡继续原RoboTwin队列。本机Goal已结束，当前无新本机作业。
