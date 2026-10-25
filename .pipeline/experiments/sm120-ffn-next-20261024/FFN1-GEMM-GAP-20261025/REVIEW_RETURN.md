# FFN1成本与MegaMoE差距核验结果

## OBSERVED

本轮没有改采用算子的计算，S源码逐字节保留。新增分项观察器＋两次活动追踪诊断，再用两次无CUPTI独立进程复测；三KV每次27tensor/七段/有效KV/输入/RNG及custom stream有限逐位通过。4296完整block调用，0 GPU失败/timeout，owned清理完成。不是机器人闭环或正式性能放行。

- S FFN1主体tile的加载/MMA/内部等待包络约140–142微秒，约占该tile146–149微秒的95.6%。它不是纯Tensor Core周期：global/shared加载、流水等待、地址计算和必要同步都在内。
- 两半GELU及输出共约5–6微秒；两次shared结果存放间隔约0.25–0.3微秒；发布约0.58微秒。浮点时间为软件观察器范围，不可当硬件逐指令counter。不同tile/warp相互重叠，不能把所有CTA时间相加为层时间。
- 阶段总包络与有任务CTA活跃包络之差约1.2–1.3微秒，包含入口/出口与尾部会合；不是纯barrier指令耗时。50个空闲CTA等待约整阶段，也不能说同步本身花掉140微秒。
- 正常完整层相对原Torch本轮1.284/1.791/2.237倍。FFN1阶段本轮0.821/0.850/0.823倍，仍落后；FFN2约1.05–1.08倍。此前多轮FFN1约0.80、FFN2约1.03，S未变；小片段相对值变化不冒称优化收益。
- 观察器整层扰动约0.25%–1.26%，local属性784→792B，255regs/shared不变。动态spill/HBM字节仍UNKNOWN。
- 当前Torch FFN1中长KV确为`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8`，grid[32,14,1]共448CTA，block128threads/4warp，158regs/thread、73728B shared。另有独立GELU kernel。名字含relu不能据此将实际模型GELU改成ReLU；数值捕获仍是原GELU。活动trace duration仅用于定位，未进入正常速度主表。short首次trace没收集kernel，保留UNKNOWN；不借中长补签。

## HYPOTHESIS / UNKNOWN

当前主因定位到GEMM主循环和配置，尚未在加载吞吐、计算指令发射、bank冲突与寄存器spill间完成硬件归因。S有120tile/170CTA，50CTA无任务，52主tile只有96/128有效行；这是确定的利用机会，不等于换成448tile就一定赢（W已反证）。不能把资源属性等同实际occupancy或把逻辑读字节当HBM流量。

## 与MegaMoE还差什么

详见[SOURCE-REVIEW](SOURCE-REVIEW.md)。我们已有persistent、Tensor Core、部分TMA和局部full/empty，但缺低成本且匹配SM120的基础GEMM、足够深的跨tile结果流水、有效按角色资源分配，以及有界跨FFN1/FFN2 ready调度。官方SM100使用UMMA/TMEM、2CTA矩阵执行及multicast，不能把我们显式DSM复制等同。官方也写ACT到global/ring、保留必要组同步；全shared与零同步不是必要条件。

尤其本轮Torch也使用warp级Tensor Core，证明不能把当前FFN1落后全归因于缺UMMA。需要先在同SM120上对齐实测库kernel的任务粒度与主循环，检查嵌入整层后的资源/调用成本。

## RECOMMENDATION与采用决定

保留S，停止扩大DSM手握块。本轮定位收口，不另开第三种设计。下一最小对照应为与实际Torch同形状/布局的4warp、K64基础主循环，先独立真实输入，再嵌入整层；保留BF16边界，不同时加DSM/专职E/跨FFN调度。只有它优于当前S且三KV完整层不退化，才推进任务ready与长流水。

完整[阶段表](STAGE-TABLE.md)、[资源/停止](RESOURCE-AND-STOP-REPORT.json)、[原始末两进程](measure-3/RESULT.json)、[第二进程](measure-4/RESULT.json)。源入口`S.cu`是未改采用源码，`whole_layer_profile.cu`仅观察器，`measure_timing.py`为最终计时。
