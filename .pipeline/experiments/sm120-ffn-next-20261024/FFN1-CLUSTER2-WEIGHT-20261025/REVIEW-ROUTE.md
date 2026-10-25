# 当前SM120：FFN1双CTA共享B无收益，保留S

最新[D结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/REVIEW_RETURN.md)、[完整18阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/STAGE-TABLE.md)、[等待定位](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/WAIT-ANALYSIS.md)、[固定源码/结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/SOURCE-AND-RESULT-ENTRY.json)。

S是实际保留的FFN两阶段版；W是上轮1T/4M/3E负结果。新L只作2CTA cluster/不同M同N配对，独立global B；新D再将B通过DSM共享，保持每CTA M64N128、48×K64完整累加、BF16/GELU、FFN2七段与阶段同步。

三KV/两个独立进程27tensor、七段、KV/输入/RNG和角色/传输计数通过。D整层比S慢60.62%/55.29%/49.89%；L比W约快1%，未自动采用L。D FFN1约558–561μs，同场W约153μs。软件观察器显示接收READY约2.79μs/K64、源槽EMPTY约3.32μs、MMA等待输入约2.9μs；这是重叠的等待区间，不是纯DSM带宽。逻辑global输入504→336MiB，同时新增168MiB DSM，不能写HBM下降1/3。

完整层原Torch/D为0.801/1.154/1.489，原Torch/S为1.287/1.793/2.232。双方eager/noGraph/warm CrossKV，阶段软件插桩与正常计时分开；本轮不是机器人E2E或独立TEST。

4836调用、17.09分钟生命周期，owned退出与15关键文件回传核验完成，公开S入口再验通过。保留所有负结果、旧FAILED与四项false。最新RoboTwin只读快照19:13为581/600组、1743/1800局，两A100继续；本机Goal原任务已结束，当前无新增本机任务。
