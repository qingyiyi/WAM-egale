# 当前讨论（2026-10-23T01:21:17.786298+08:00）

用户要求优先Self QKV，核查为何W3–W6 MMA、能否全8warp或更大tile，并再次要求同步新版完整阶段表。通过Maintain更新当前M39交付约束，不把更大tile提问写成决定。

已实际比较固定CTA64×128下4→8warp。共用函数混合分支使原4warp阶段也变慢；隔离8warp函数后当前Q/K/V约快10%–15%，整层同轮快约1.4%–1.9%，采用。不是只删warp判断，实际warp tile32×64→32×32，迭代器/同步/写回映射一起匹配。编译栈帧明显降低支持函数上下文影响，但远端无counter，未精确分摊动态spill或Tensor Core利用率。

新完整阶段表全部重测：当前Self QKV约0.77–0.85×Torch，其余三投影约0.65–0.72×，FFN区间约1.02×。建议下一步优先同形状Self-out/Cross-Q/out适用性，尚未运行；更大tile和联合QKV任务池未实现，不预设最佳。baseline/投机/旧tile/保护文件未动，4科学权限false。

GPU0当前清理，累计16672.587/36000秒。GPU1 RoboTwin固定队列19/1800继续；microwave valid000三臂均失败保留。用户要求每次采用优化后同步整张阶段表，含tile工作量、warp、同步和原Torch加速比。

[新版阶段表](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/STAGE-TABLE.md)、[本轮结论](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESULT.md)。
