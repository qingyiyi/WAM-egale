# SM120 TMA与FFN2粒度有界实验回传

完成时间：2026-10-23T22:25:11.629514+08:00。当前保留GRAIN-M64-FFN2研究入口，TMA退化不采用；旧A与原始失败/无效记录均保留。只改单层独立SM120 optimized工作区，未修改baseline、生产投机入口、保护state/cycle/rules或RoboTwin队列。

## OBSERVED：两项实际候选

|候选|改动|三KV逐位/状态|相对同轮A的短/中/长加速比|结论|
|---|---|---|---|---|
|TMA|FFN2二维tensor map、128B swizzle、两槽K64异步搬运；保留48输出tile、四MMA warp、七段K2048|通过|0.9450× / 0.9506× / 0.9523×|完整层退化5.0%–5.8%，不采用|
|GRAIN|原cp.async，FFN2输出tile M128→M64，48→96任务；保留四MMA warp、七段K2048及BF16提交顺序|通过|1.2246× / 1.2048× / 1.1892×|研究采用|

GRAIN完整层延迟下降：18.34% / 17.00% / 15.91%。两独立进程均改善，自有24轮中位数的IQR/median为0.216%–0.998%。时间比例按两进程中位数均值计算，不挑最快轮。

|真实KV|A ms|GRAIN ms|原Torch/GRAIN|guarded/GRAIN|GRAIN比guarded延迟高|
|---|---:|---:|---:|---:|---:|
|short|0.976218|0.797178|1.0864×|0.9569×|4.50%|
|middle|1.042075|0.864910|1.5430×|0.9491×|5.36%|
|long|1.123172|0.944474|1.9498×|0.9274×|7.83%|

四臂同卡、同输入、同缓存、全eager无Graph。无插桩正常二进制决定完整层结果；软件阶段插桩仅用于诊断。完整融合表见[STAGE-TABLE](STAGE-TABLE.md)，未将FFN1/2重叠区拆成可加时间。Cold CrossKV与原Torch调制表单列。

## 精度与实现

三份已有真实B14/Q16、KV288/1152/2160、block0输入：26项中间量与最终输出逐位一致；有效KV、输入、RNG不变。两进程计时前后输出核验、独立public默认流/非默认流完整调用也通过。FFN2四组任务领取/完成各24，原FFN1计数28/28/28/36。仅有限fixture，不证明任意输入、30层或闭环机器人质量。

当前整层170CTA×256线程，动态shared100096字节、255寄存器/线程、静态localSize784字节。TMA版shared100352字节；均为每SM一个CTA。static localSize不能代替运行时local流量；未访问受限NCU计数器。当前仍使用warp级BF16 HMMA，未引入SM100专用UMMA/TMEM。

- 当前源码：`../workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`。
- `ffn2_partition64`（344行）：64×128 tile，四个MMA warp，保持七个K2048分段。
- `ffn2_commit64`（376行）：八warp四元素向量写回；原BF16舍入、最后bias、门控残差顺序不变。
- `action_ffn_pipeline`（585行）：原两个生产行块按相应就绪计数服务四个M64消费行块；只改变输出任务粒度，没有扩展完整DAG或改变阶段grid同步。
- `whole_layer.py`与构建脚本显式使用唯一`wam_whole_grain_cuda`模块，避免与冻结A扩展混载。
- TMA源：`TMA-source/csrc/whole_layer.cu`、`tma_ffn2.cuh`；设备反汇编确有`UTMALDG.2D`和BF16 HMMA，见`TMA-INSTRUCTION-EVIDENCE.json`。这些是静态指令证据，不是运行时流量/重叠统计。

## HYPOTHESIS / UNKNOWN

TMA实现每K64步引入两次CTA barrier、mbarrier等待，双槽及fragment预取方式也与原CUTLASS不同；整体FFN从历史约0.51ms升至本轮约0.57ms。软件阶段支持退化集中在FFN，尚未隔离同步、TMA吞吐、fragment流水、host tensor-map编码的各自贡献。不能宣称TMA在5090普遍无效。

M64把潜在并行输出任务从48增至96，同时每任务累加器/计算工作减半；也改变了warp tile和权重请求复用。结果支持针对170SM调整并行粒度有益，不能仅凭本实验声称SM利用率翻倍或HBM流量减少。未测实际CTA并发峰值、带宽、L2/DRAM和Tensor Core活跃率。

这次仍未超过强guarded基线，不能宣称新增调度已赢过最佳B14路径。四科学权限及production_admitted仍false；没有新闭环、TEST、完整chunk或任务E2E结果。

## 失效记录与止损

首轮`TMA-measure1/2`同名扩展复用A，候选正常耗时无效；原记录不改，并用独立模块名修正重测。它们的调用/时间全部计预算，不能引用其原COMPLETE字段作为候选性能通过。详见`MEASUREMENT-IDENTITY-CORRECTION.json`。两次只读工具查找错误（远端无cuobjdump、未激活环境时无python）与一次文件未传完的本地反汇编读取错误已改用实际工具/完整文件；未发生GPU超时或新增数学失败。

仅完成预登记两个候选，不继续搜索tile网格、不把TMA与M64再组合。推荐保留M64，下一轮优先分清剩余FFN2计算/七段提交开销，再考虑三个慢小投影；不自动新增GPU实验。

## 资源与固定入口

见[SELECTED.json](SELECTED.json)、[源回传](SOURCE-RETURN-MANIFEST.json)、[重复测量审计](PERFORMANCE-REPEAT-AUDIT.json)、[资源账本](RESOURCE-AND-STOP-REPORT.json)。源码、二进制与原始结果已回本机并逐文件byte核对；未做SHA。自有进程自然退出，5090可见compute为空、显存0MiB；实例没有关机。两张A100原RoboTwin队列未动。

资料依据见[REFERENCES](REFERENCES.md)，实际CUDA13.0.88/Torch2.9.0+cu130/CUTLASS3.5.1；没有升级依赖或修改驱动。
