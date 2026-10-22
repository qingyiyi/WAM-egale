# 当前执行结果：独立8warp Self QKV

2026-10-23T01:26:04.376194+08:00。当前已采用独立Self QKV设备函数，8warp MMA，CTA64×128、warp32×32；没有扩大tile。原4warp其他三投影保持。首次混合函数使完整层慢1.72%–2.02%，隔离后同轮完整层延迟降1.769%/1.819%/1.438%；两版各9真实/36结构/225次有限门通过。负结果、首次CPU编译失败均保留。

当前原Torch/自有完整单层warm Graph：**1.2661/1.8074/2.9322×**（KV288/1152/2160），双方同Cross最终KV预计算，原Torch静态dense history；不是30层、chunk、机器人E2E或guarded增量。新完整阶段表全部重新测量，Self QKV约0.77–0.85×，仍未追平原Torch；其他三投影约0.65–0.72×，FFN共同区间约1.02×。阶段边界不同于独立原语，不能直接相加。

栈帧证据：原4warp函数208B，混合4/8函数256B，隔离8warp函数8B；静态字段不能代替动态HBM或利用率。远端无GPU counter访问，不再尝试提权或改驱动。下一步建议同形状Self-out/Cross-Q/out有限核对，尚未运行；没有放大tile、没有重新开启全层ready。

两机源码ce9d30a4…、DSO4f041cdd…，01:21全部owned退出。累计16672.587/36000秒，GPU0空闲。GPU1 RoboTwin自动队列19/1800；T6/7、C5/6、P5/6，6组配对审计通过，microwave valid000三臂均失败保留。这是扩大DEV早期状态，不是总体SR或独立TEST。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESULT.md)、[新版完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/STAGE-TABLE.md)、[精确身份](../.pipeline/experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/SELECTED.json)、[资源](../.pipeline/experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESOURCE-AND-STOP-REPORT.json)、[HW入口](../.pipeline/PROGRESS.compact.md)。四科学权限及production_admitted均false，旧review/RESULTS.json未改。
