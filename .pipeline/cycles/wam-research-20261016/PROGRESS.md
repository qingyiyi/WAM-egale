---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-23T01:21:17.786298+08:00'
current: M39
next: 根据新版阶段表讨论Self输出与Cross Q/out同形状投影；GPU1固定机器人队列继续，GPU0已清理。
---

# 当前研究进度（2026-10-23T01:21:17.786298+08:00）

M39当前采用独立8warp Self QKV：CTA仍64×128/K64，warp32×32，W0–W7均cp.async/MMA，原两段K1536与BF16/bias舍入保留；96tile/投影、108CTA、27grid同步、3408/3480/3408总tile不变。Self-out/Cross-Q/out仍原4warp。两机源码ce9d30a4…、DSO4f041cdd…；完整身份见SELECTED。

首版共用函数混合4/8warp：有限门通过但整层退化1.72%–2.02%，负结果保留。拆成独立设备函数后同轮完整层延迟降1.769%/1.819%/1.438%，两个进程均正向。两版各9真实签名/36结构控制、225次有限完整调用逐位/状态通过；不证明所有输入。当前原Torch/自有完整单层warm Graph **1.2661/1.8074/2.9322×**，双方同最终Cross K/RMS与V预计算、Torch静态dense history；不是最强guarded新增收益、30层或机器人E2E。

当前完整阶段表已全部重测，两进程测量副本资源同；Self Q/K/V对原Torch约0.77/0.83–0.84/0.82–0.85×，仍未追平，但比同轮4warp对应阶段约快10%–15%。其余三小投影约0.65–0.72×，FFN共同区间约1.02×。阶段为核内区间对Torch独立热图，缓存上下文不同，不能相加分账。Self QKV设备函数静态栈帧8B、原函数208B；混合版256B。支持编译上下文影响，不把静态spill当动态HBM或将收益全归Tensor Core利用率。远端硬件计数仍不可访问。

本轮CPU编译失败1次、0GPU调用，原记录保留；后续门和计时均完成。GPU workers228.965秒，CPU编译620.624秒，保守含准备/等待/清理1356.704秒，累计 **16672.587/36000秒**，剩余5.369小时。01:21已清理全部owned，GPU0空闲；本轮01:28:41与此前截止不改。本机0GPU，远端0NCU。四科学权限及production_admitted仍false。

M40截至2026-10-23T01:18:50.673113+08:00：GPU1固定1800局扩大DEV完成19局；T6/7、C5/6、P5/6成功，6组配对审计通过。open_microwave/valid000三臂均完整失败，全部保留；当前place_object_basket-valid001-C。队列自动接续，GPU0固定留算子；优化reset bank沿用。旧24小时账本不改，新160GPU小时/最长10天、T0=10/22 22:03:22；这不是独立TEST或总体SR结论。

|ID|阶段|状态|当前证据|下一步|
|---|---|---|---|---|
|`M39`|单层算子|in_progress|独立8warp Self QKV采用，新完整阶段表|讨论其余同形状三投影，暂不扩大tile或全层ready|
|`M17`|LIBERO证据|blocked|四套6000局齐，独立性/配对/长尾缺口保留|不恢复过期队列|
|`M40`|RoboTwin扩大DEV|in_progress|GPU1自动队列19/1800|定期核验完整终态与自动接续|

[本轮结果](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESULT.md)、[新版完整阶段表](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/STAGE-TABLE.md)、[资源](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESOURCE-AND-STOP-REPORT.json)、[RoboTwin检查点](../../experiments/robotwin-pilot-20261020/eval100-20261022/CHECKPOINT-20261023-0119.json)。10/23冻结身份与claim、10/28主表、11/01 18:00证据包；不足则收缩claim。
