# 既有FFN负结果证据矩阵

固定基线均为当前SELF-QKV-TMA-4MMA A；原始证据在../rtx5090-block-port-20261023/ffn-local-handoff-20261024/。不重跑这些候选。下列比值是候选耗时/A，短中长KV顺序；局部为同轮profile区间，整层为两个独立进程无插桩结果，不能互相替代。

|候选|tile/warp/任务|FFN1重算|有限数值/状态|FFN/完整block相对A|资源/流量证据|已证与未知|
|---|---|---|---|---|---|---|
|B专职TMA输入|M64/N128；96×7=672；1TMA/4MMA/8warp有序写回|1×|三KV逐位/KV/RNG/计数通过|FFN1.0276/1.0345/1.0305；block1.0120/1.0113/1.0083|静态资源同A，双输入缓冲|分工可运行但无收益；没有实际stall/DRAM归因|
|C跨任务写回重叠|同672；1TMA/4MMA/3epilogue；双输入/元数据+单FP32结果槽|1×|三KV有限逐位/状态通过|FFN1.0155/1.0131/1.0084；block1.0090/1.0060/1.0072|静态资源同A，真实完成计数通过|没有量化真实重叠时长，不能把意图当吞吐收益|
|D小块ACT驻留|M16/H2048=64KiB；98驻留任务；336输出tile×7|1×，不是重算|三KV有限逐位/状态通过；首次模板编译失败保留|FFN4.0056/3.9792/3.9684；block2.2556/2.1461/2.0332|FFN1逻辑权重tile请求180→1176MiB；FFN2 336→1176MiB|M变小导致权重跨行复用下降可由源码确认；不是实测DRAM带宽根因|
|E双CTA DSM|170CTA/85cluster；rank0生产ACT，两CTA各消费12列tile，前后cluster.sync|1×|真实cluster launch/远程shared及三KV有限逐位通过|FFN6.0609/6.0344/5.9867；block3.1233/2.9319/2.7451|85组两波做98任务；cluster布局改变其他阶段放置|不是仅DSM一个变量，不外推DSM无效或不支持|

原始完整结果：各候选results/measure{1,2}-data/RESULT.json；局部同轮final-comparison/data/RESULT.json。对应源码whole_layer.cu、CHANGE.patch、SOURCE-ENTRY.json、资源/清理收据保留。D原模板失败与r1分别记录。

现A保留global ACT供24输出列tile共享；唯一ACT6.125MiB，一次写一次读12.25MiB只是唯一数据逻辑下界，不是RTX5090 GDDR7 DRAM实测。实际跨CTA读放大、L2命中和权重/partial流量未分离。实测L2容量96MiB、每CTA opt-in shared上限99KiB，不据容量推断命中。没有可用硬件计数器证据，静态local与动态spill分列UNKNOWN。

本轮相邻K段同owner与B/C/D/E不同：不改ACT、FFN1、M/N、MMA分工或驻留规模，只研究FFN2 partial状态的组内归属、任务数和发布。它可能减少split-K并行，所以不能预设比已有A快。
