# FFN交接与ACT驻留：最终有限结果

## 结论

OBSERVED：B输入专职、C跨任务写回重叠、D单CTA ACT驻留、E双CTA DSM共享均已实际编译和完成真实三KV有限数值/状态验证、两个独立进程四臂正常整层测量。四候选均未得到可保留收益，**全部不采用，主实现A保持字节不变**。

|候选|FFN局部耗时相对A：短/中/长|正常整层耗时相对A：短/中/长|决定|
|---|---|---|---|
|B FFN-B-INPUT|1.0276/1.0345/1.0305×|1.0120/1.0113/1.0083×|不采用|
|C FFN-C-OVERLAP|1.0155/1.0131/1.0084×|1.0090/1.0060/1.0072×|不采用|
|D D-ACT-RESIDENT|4.0056/3.9792/3.9684×|2.2556/2.1461/2.0332×|不采用|
|E E-CLUSTER-PAIR|6.0609/6.0344/5.9867×|3.1233/2.9319/2.7451×|不采用|

所有加速比同时保留原Torch/Guarded强对照，全部eager、无CUDA Graph、相同CrossKV缓存。FFN局部来自最后同轮交错的profile DSO，完整层来自每候选两进程无插桩测量；不混用边界、不把局部结果乘成E2E。

## 实际实现

- B保留96输出tile×7K2048=672任务、四MMA和八warp有序写回，只分出W0 TMA及双输入缓冲。
- C同672任务、七段舍入，一TMA/三epilogue/四MMA；双输入/双元数据槽＋单FP32结果槽，MMA可在寄存器计算下一任务、等待写回释放结果槽后再存。不扩完整DAG。完成标记校验通过，但没有对真实重叠时间作定量测量。
- D用M16×K2048=64KiB ACT驻留，完整16个FFN1 H128 tile产生后消费24个FFN2 N128 tile。98任务、336输出tile×7提交；FFN1不重复计算，无新FP32全局中间量。旧ACT workspace暂留但正常路径不写/read，不能宣称显存下降。
- E是实际cudaLaunchKernelEx cooperative cluster2＋cluster.map_shared_rank DSM；170CTA=85组。rank0生成ACT，两CTA分别消费12列tile，前后cluster.sync保护发布和生命周期。三KV中间量/输出有限逐位及有效KV/RNG通过，**证明这份真实kernel可执行，不证明双CTA布局高效或适用于全部输入**。全kernel cluster launch也改变其他阶段放置，E/D不是仅有DSM一个变量。

D首次整体device Gemm模板因M16线程映射静态断言编译失败，0GPU调用；原记录保留。r1改直接warp级DefaultMmaTensorOp，不改变数值标准。实际源以各目录whole_layer.cu和修正CHANGE.patch为准；早期prepare脚本不是最终源码入口。

## 为什么ACT驻留没有获益：可核对结构代价

以下是依据源码tile循环推导的**逻辑权重tile读取字节**，不是实测HBM流量：

|项目|保留A|D/E驻留|变化|
|---|---:|---:|---:|
|FFN1权重tile逻辑请求|180MiB|1176MiB|6.533×|
|FFN2权重tile逻辑请求|336MiB|1176MiB|3.5×|
|FFN1输出tile数|120（108主＋12尾）|1568|13.07×|
|FFN2部分和tile数|672（M64）|2352（M16）|3.5×|

固定相同数学，放下完整K2048 ACT迫使M缩小。FFN1/FFN2的权重跨行复用下降；D只有98个驻留任务能工作，其余72个常驻CTA等汇合。E生成ACT时每组只有rank0计算，85组分两波完成98任务。双CTA还需DSM读取和cluster同步。这些结构变化可从代码确定；本实验没有把每个因素的耗时单独分离，不能宣称某个因素解释全部退化。

运行时查询L2为96MiB、opt-in shared上限101376B。完整ACT仅6.125MiB，但**实际是否驻留L2未知**，不得把容量可容纳当cache-hit实测。新原型省的是全局ACT接口访问，不代表省同量HBM流量。MegaMoE固定参考也使用全局l2_acts环：L1输出与L2输入descriptor绑定同一global tensor；不是把整个中间层留在一对CTA shared。

B/C静态资源与A相同（见RESULTS.json），仍慢约1%，不能把当前负结果简单归因新增shared导致occupancy下降。没有NCU权限，动态访存、stall、真实TMA/MMA重叠均UNKNOWN。

## 质量与边界

27捕获中间量/最终输出、normal/debug、输入/全部有效KV、CPU/CUDA RNG和各新schema覆盖三份实际block0 fixture。没有重新验证其他层、在线接受分数/路径、完整模型或机器人；四科学门及production_admitted=false。当前A不变，沿用原公开默认/非默认流证据，不把新候选借用其身份。

## 建议与止损

本轮停止这四个版本的投入，保留所有失败与负结果。下一步保留较大M和ACT全局共享，优先单独归因FFN1尾部/写回与K片段ready门；不要继续通过压小M强留完整ACT。任何新实现仍需单因素、精度和正常整层对照，不保证下一方案会更快。

结果入口：RESULTS.json；最终当前A完整阶段表STAGE-TABLE.md；各候选STAGE-TABLE.md与snapshots；原始完整计时results/measure{1,2}-data/RESULT.json；同轮局部归因final-comparison/data/RESULT.json。

资源：所有作业生命周期之和1207.323秒，编译1095.558秒，GPU工作进程生命周期合计108.073秒，CUDA active UNKNOWN；13380次完整block，1次编译失败、0 timeout。33个自有PID+出生身份及阶段进程组均退出，5090 context为空、实例未关机。旧账本保留、后续累计5090按新授权无旧上限；双A100原预算不改。详见RESOURCE-AND-STOP-REPORT.json。
