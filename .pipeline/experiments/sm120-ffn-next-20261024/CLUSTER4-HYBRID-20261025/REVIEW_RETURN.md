# 四CTA DSM/global混合FFN：返回审查

采用决定：**NO_GO，保留原A（SELF-QKV-TMA-4MMA、170CTA）**。B4/C4均不集成机器人模型。

## OBSERVED

三档真实B14输入（短/中/长KV），27个所存中间量、七个K2048分段的partial/running BF16递推、有效KV、输入未修改和RNG检查均有限通过。默认/custom stream均覆盖；有限fixture通过不证明所有输入逐位等价。

A为原独立二进制；B4只将原设备体改用cluster4启动；C4保持原FFN1/global ACT及七段BF16写回，四CTA共享局部ACT分片。实际cluster容量41组=164CTA，低于A的170CTA。每CTA仍8warp、100352B shared、255regs/thread、784B静态local；插桩版本local792B。

|KV|Torch ms|Guarded eager ms|A ms|B4 ms|C4 ms|C4比A增加|
|---|---:|---:|---:|---:|---:|---:|
|short|0.869341|0.777741|0.693606|0.725280|0.978848|41.12%|
|middle|1.342138|0.829270|0.764362|0.879286|1.132064|48.11%|
|long|1.850403|0.885258|0.846502|0.879328|1.133200|33.87%|

以上为最终c4-pre-gate同一进程独立无插桩整层计时：warm CrossKV、同卡、eager、双方无CUDA Graph。没有机器人E2E或正式质量结论。另一独立C4进程也出现同方向约34%–48%整层退化。完整阶段表见[STAGE-TABLE.md](STAGE-TABLE.md)。

B4的FFN约0.296–0.298ms；C4约0.548–0.550ms，新增约0.25ms主要落在FFN路径。B4相对A还存在cluster启动/164CTA的退化；168个attention任务跨过164驻留边界，与额外调度波次一致。阶段插桩不能当独立无插桩延迟相加。本轮没有单独测得无插桩C4 FFN子图延迟；完整层在独立进程重复退化已足以止损，因此没有追加该项，也不将阶段计时冒充无插桩FFN结果。

## 实现与数值顺序

- FFN1仍120任务，global ACT仍保留。FFN2仍96个输出tile×7个K2048分段=672个CTA任务。
- C4变成168个cluster job，每组4CTA处理不同N128输出；每CTA持有M64×K128 BF16=16KiB ACT，共享为K512。每个K2048分段经过4个K512片。
- DSM普通向量读把其他CTA的ACT搬入本CTA CUTLASS swizzled A操作数；W2按K32循环与原K16 MMA归约次序处理。四MMA warp3–6不变；没有新增专职TMA warp。
- 每片READY和读者完成使用cluster同步，原block阶段grid同步保留。global F2 partial及每段BF16舍入不变，没有改成七段连续FP32只舍入一次。
- 相同组四个输出tile的重复ACT global逻辑读取可减少75%；这不是实测HBM节省。跨cluster仍走global，符合本次混合路径要求。

## HYPOTHESIS / UNKNOWN

B4→C4同时改变DSM共享、交接同步、操作数布置与K32/W2加载循环。观察支持“整个混合FFN路径变慢”，不能分解成DSM本身的精确耗时。164CTA造成额外attention波次有源码与对照支持；真实L2/HBM流量、动态spill和硬件stall未知，未使用受限GPU计数器。没有出现shared资源减少或驻留提升。

## 检查顺序与保留的历史

初始c4-1/B4脚本在计时前核对完整输出/27中间量，但七段和状态专项核验位于计时后；该顺序不足以满足全部先正确性再计时要求，原数据保留且明确标注。后续将七段/有效KV/输入/RNG门移到计时前；c4-2与最终c4-pre-gate均完成前置核验后计时。最终主表取c4-pre-gate，未改候选设备数学。不是新增设计搜索。

## RECOMMENDATION

停止四CTA混合候选，保留当前较快global A。当前结果说明本实现交接代价大于省去重复ACT读的收益，不证明DSM普遍无效。后续优先分解原A FFN与Torch仍约15%的阶段差距；本轮不自动新增2CTA、扩大tile或另做完整DAG。

## 固定入口与资源

- 当前保留源：`../../rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`（相对路径导航请以SOURCE-AND-RESULT-ENTRY.json的绝对路径为准）。
- 本轮独立候选：`B4.cu`、`C4.cu`，构建脚本`build_b4.py`/`build_c4.py`；插桩源单独保存。
- 原始输出及独立二进制：`remote-return/results/`、`remote-return/build/`。最新归档状态以`RESOURCE-AND-STOP-REPORT.json`为准。
- 累计7122整层调用，8个编译/测量阶段真实exit0，未发生GPU失败或超时；完整生命周期、回传与进程身份清理收据单列。
- 本机M42仍是Goal接受阈值DEV；本算子结果不改变四项false、生产tau、selector、旧FAILED或质量证书。


归档已闭合（2026-10-25T14:18:28.699122+08:00）：56文件/369215507字节回传、尺寸与fsync核验；完整生命周期28.98分钟，GPU active未知，7122调用。8个阶段真实exit0，owned PID/birth已退出，A源码未变。见RESOURCE-AND-STOP-REPORT.json。
