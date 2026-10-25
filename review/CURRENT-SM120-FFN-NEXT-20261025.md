# 当前SM120单层：双CTA仍不采用，保留A

最新双CTA测量已闭合：[结果](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)、[全17阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/STAGE-TABLE.md)、[逐进程原始标量](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/NORMAL-RESULT-TABLE.md)、[资源与停止](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/RESOURCE-AND-STOP-REPORT.md)。

B2只改cluster2启动几乎不变；C2比同进程C4快1.19–1.30倍，但仍慢原A12.27%–15.46%，G2绕过peer DSM仍慢8.71%–10.42%。R0/R1观察器支持输入就绪/有序提交进度开销；无计数器，不能称DSM raw load或HBM根因。当前实际采用SELF-QKV-TMA-4MMA A170不变。

新正常进程原始RESULT放本目录raw-normal；服务器完整数值trace/DSO不上传。SOURCE入口的本机/远端绝对路径仅定位用，不声称网页可独立复算。旧负结果与四false保留，未集成机器人路径。
