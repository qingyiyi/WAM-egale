# 当前SM120：FFN1双K64交接A2负结果，保留S

最新[结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/REVIEW_RETURN.md)、[完整18阶段](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/STAGE-TABLE.md)、[冻结候选源码](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/whole_layer.cu)、[原始进程1](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/measure-1/RESULT.json)、[进程2](../.pipeline/experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/measure-2/RESULT.json)。

本轮仅FFN1＋GELU输入交接：两个相邻K64的8KiB A合为一次16KiB DSM，48次credit/ACK降为24次。tile/1T4M3E/FP32累计/BF16 bias与GELU/FFN2七段均不变。三真实KV、两进程、27张量/七段partial-running/有效KV/输入/RNG与计数有限逐位通过。

A2完整层相对旧共享A慢4.35%/3.86%/3.44%，相对S慢31.73%/28.77%/25.92%，不采用。当前保留S同轮原Torch/S=1.286/1.789/2.229×，eager/noGraph/warmCrossKV；不是机器人E2E。A2 FFN1约355μs、同场W约152μs。相邻交接合并失去step+2提前装载，结果与该代价一致，未独立量化等待/带宽，不宣称HBM节省。4842调用、11.62分钟生命周期，owned退出与IO恢复闭合。

原S/W/共享B/共享A/DSM负结果均保留；本轮未进一步搜索、未改生产入口。下一步建议先对齐S的FFN1 GEMM，而非盲增DSM传输块。

20:48双A100原队列598/600完整三臂组、1794局，T/C/P成功487/485/491，两worker无可领取任务后退出；旧281/406两缺口不自动重试。5090 RoboTwin已开始完整DEV：首个篮子valid49 Target完整失败，OLD接续；尚无四臂组，不能确定最终tau；原21:56:53截止不自动延长。本机无自有GPU作业。四项科学权限false、旧FAILED/retention/FIT-DEV暴露不变。此页是发布时快照。
