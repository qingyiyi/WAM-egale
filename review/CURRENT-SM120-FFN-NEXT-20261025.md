# 当前SM120单层：采用FFN两阶段同步版S

[本轮结果](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)、[完整18阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/STAGE-TABLE.md)、[冻结源码](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/whole_layer.cu)、[原始标量](../.pipeline/experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/SUMMARY.json)。

FFN1+GELU写global ACT→grid同步→FFN2七段K2048→最后一段门控/残差。只增加阶段边界，tile/warp与BF16顺序不变。三真实KV、27tensor/七段/状态/公开入口有限逐位通过，两个独立正常进程测得原Torch/S约1.288/1.797/2.238倍，相对原A延迟下降2.25%/1.82%/1.90%。双方eager/noGraph/warm CrossKV，限单层，不是机器人E2E。

FFN1+GELU约0.80×Torch是当前优先定位项；FFN2+残差约1.03×。不能把这次改善直接归因于某一种等待或缓存机制；未测硬件流量。2865调用、17.12分钟生命周期，owned清理完成。

[历史C2/G2负结果](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)保留，原A独立保存，未重算原FAILED。网页可读源/标量；原始tensor、DSO、权重不公开。四项科学权限仍false。
