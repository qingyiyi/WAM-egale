# Self QKV：增加MMA warp后的函数隔离结果

2026-10-23T01:21:17.786298+08:00。采用独立8warp Self QKV；没有放大CTA tile、没有改变候选/数值规则。完整层比同轮4warp控制延迟下降短1.769%、中1.819%、长1.438%；两个进程各档均正向。原Torch/当前完整层 **1.2661× / 1.8074× / 2.9322×**。仍是有限真实输入单层warm测量，四科学权限与production_admitted均false。

## OBSERVED

原4warp输出分块：CTA64×128，warp32×64，因此2×2=4个MMA warp，映射W3–W6是旧实现的编号约定。它们同时做cp.async和MMA；W0/1/2/7在主循环等待，所有8warp再协同写回。W0线程0在tile边界领取任务，不是持续运行的专用生产者。

本轮固定CTA64×128、K64/3stage、96输出tile、两个K1536分段与原BF16/bias顺序，把warp输出改32×32，2×4=8warp。总数学工作不变。每线程accumulator理论元素64→32；这不是总MMA吞吐翻倍。

第一版在共用action_projection中混合4/8warp分支：9真实签名/36结构控制、225次有限完整调用通过，但完整层慢1.72%–2.02%。两进程软件时间线发现Q/K/V慢约3–4μs，仍为4warp的Self-out/Cross-Q/out也慢约2–5μs。该函数ptxas栈帧208→256B，spill stores/loads各204→252B。测量副本资源相同、扰动≤0.239%。原始负结果保留在r2与profile。

第二版把8warp Self QKV放进独立设备函数，原4warp action_projection源码恢复。同样9签名/36结构/225次有限门通过。ptxas原函数恢复208B栈帧/204B spill，新self_qkv_projection8仅8B栈帧/spill。整层资源仍108CTA、112384B shared、255寄存器、1016B静态local、1CTA/SM。两进程phase控制显示Self Q约51–52→45μs，K约47–48→41–42μs，V约47–48→41–42μs；其他投影不再出现首版2–5μs一致退化。phase副本扰动≤0.289%，有限状态通过。

最终所有阶段对原Torch重新测量，不沿用旧阶段耗时；见[新版完整阶段表](STAGE-TABLE.md)。第一轮CPU构建因源码生成漏函数体失败，0GPU调用，已保留；修复在同一截止内新attempt完成。后续GPU门及计时无失败/timeout。

## 形状与选择

|配置|CTA输出|warp输出|MMA warp|每投影输出tile|本轮结论|
|---|---|---|---:|---:|---|
|原控制|64×128|32×64|4|96|保留为证据|
|混合函数8warp|64×128|32×32|8|96|整层退化，不采用|
|独立函数8warp|64×128|32×32|8|96|有限门通过、整层改善，采用|
|更大CTA候选|128×128或64×256|需单独选择|可8|48|仅形状推导，未实现/未计时|

M224,N3072下当前tile数ceil(224/64)×(3072/128)=96；扩大输出tile使任务减到48，可能增加权重复用但降低同时可执行CTA。Q/K/V虽独立，目前仍按三个阶段执行；合并任务池是另一个变量，未实现。不能把任务总数凑108当优化目标。

## HYPOTHESIS / UNKNOWN

对照支持函数组织/编译上下文是重要因素，不能把首版退化归因为8warp天然更慢，也不能把新版收益全归为Tensor Core利用率增加。更小warp tile的accumulator减少与设备函数保存/恢复减少一致，但没有远端硬件计数器，动态local流量、流水stall、各机制耗时占比仍UNKNOWN。静态栈/spill字节不能当实际HBM字节。8warp不是全形状最优保证。

## RECOMMENDATION

当前保留独立8warp Self QKV，下一步优先对同形状的Self输出、Cross Q/out做同样有限核对，先处理剩余小投影成本，再考虑更大tile或QKV联合任务池。后一轮尚未启动。本次最佳仅指已测当前版本和有限形状。

分块与寄存器复用的背景见[NVIDIA GEMM性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)、[CUTLASS GEMM层次与流水](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)。这些解释取舍，不代替本项目实测。

[完整层原始摘要](SUMMARY.json)、[首版阶段](PHASE-SUMMARY.json)、[最终阶段](CURRENT-STAGE-SUMMARY.json)、[身份](SELECTED.json)、[资源与清理](RESOURCE-AND-STOP-REPORT.json)。新增GPU worker生命周期228.965秒，CPU编译620.624秒；保守总计1356.704秒，累计16672.587/36000秒。所有owned退出、GPU0空闲，GPU1机器人另行继续。
