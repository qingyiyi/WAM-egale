# 单层GEMM双缓冲研究入口

主算法：[csrc/gemm_pipeline.cuh](csrc/gemm_pipeline.cuh)。独立封装：[whole_layer_pipeline.py](whole_layer_pipeline.py)。构建：[build_pipeline.py](build_pipeline.py) / [pipeline_codegen.py](pipeline_codegen.py)。旧whole_layer.cu、旧DSO和baseline保持，供教学与同输入对照；本入口未接入模型默认路径。

只替换完整单层中的10个GEMM阶段的数据搬运/同步，数学仍64×64×32、按K递增、原WMMA累加顺序及BF16 bias写回。attention、norm、RoPE、scheduler、cache与粗stage barrier保持。独立编译生成sync/async/profile三个唯一namespace；sync为同LD40布局对照，profile为async加阶段计时，不是三个生产候选。

## 数据所有权与交接

- warp0领取CTA tile；warp1负责A，warp2负责W；warp3–6计算；warp7写回。
- 共享内存有两组A/W buffer，每行40个BF16，其中前32为有效K。每次拷贝16字节，loader lane各有不重叠的地址；无效M/N行使用有效global指针和src-size=0做零填充。
- prologue加载buffer0，loader执行cp.async.wait_group 0，然后所有线程CTA barrier，消费者才可读取。
- 第z轮消费者读current，loader往另一个buffer发起下一块拷贝。loader等自己的copy group完成；本轮末尾CTA barrier同时保证下一块ready和current不再被读取。下一轮才可复用旧buffer。
- 最后一次copy已被wait_group排空，才进入原epilogue；没有把同一shared地址同时用作未完成copy和消费者工作区。

这利用了[官方PTX cp.async/commit/wait语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)，只使用SM80指令。wait_group是每线程copy完成等待，随后CTA barrier负责跨warp交接；不使用Hopper专有TMA/WGMMA。

原GEMM的每K块“加载→barrier→MMA→barrier”变成“下一块异步加载与当前MMA→交接barrier”。K3072的内部CTA barrier从194变99，K14336从898变451，另有任务领取/完成与stage同步。这是静态同步次数，不等于实测stall下降比例。

仍然是阶段内流水，尚无跨stage ready队列、跨CTA普通shared memory或candidate query合并。实测async172 registers/thread、59616B shared、理论1 CTA/SM；profile180 registers。理论驻留不是achieved occupancy。

## 验证边界

M39先做9真实签名、18个合法KV状态转移及输入/RNG/参数门，再做两进程完整block计时与独立profile。匹配旧失败whole仅说明本次搬运改动未引入新的有限样本差异，不能消除旧Torch数值FAIL或证明所有输入等价。数据见[本轮登记](../../../.pipeline/experiments/single-layer-async-gemm-20261020/PLAN.md)，最终以RESULT及真实终态为准。

## 本轮结果与后续切块

[完整证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/RESULT.md)：两独立进程9点432样本，对旧原型耗时下降17.9%–36.3%，仍比native慢3.33–7.84倍。9点/18合法KV转移匹配旧失败原型；原Torch逐位9/9失败，没有生产准入。登记上限240/900不一致作为资源审计FAILED保留，GPU已清理。

当前GEMM64×64×32、每计算warp32×32，WMMA16×16×16；attention Q16/KV64/head128；Norm完整行3072；pointwise128元素。可先评估GEMM K64以减少循环/同步，再按M/N实际大小考虑更宽N；B2 action M32扩大M会浪费。长KV attention的加载/softmax仍是独立瓶颈，不能直接把不同candidate拼成更长query序列。这些更大tile尚未实现或测量。

用户最新要求后续直接改optimized，不再为另一agent带读保留独立版本或新建实验登记；此前对照源码身份仍可通过实验manifest和Git追溯。

## 与FlashAttention-2的范围差异

[FA2论文](https://arxiv.org/abs/2307.08691)优化attention本身的线程块/warp分工、非矩阵运算和shared通信；不包含本模型整层的QKV投影、FFN、Norm与残差。本实现把这些阶段都放进同一个cooperative kernel，但仍按整阶段grid barrier推进，中间tensor保存在global workspace。单kernel结构不保证比成熟多kernel路径更快。

本attention在同一candidate/head内按Q16/KV64做QK→在线softmax→PV，不落完整Q×KV分数矩阵；但softmax目前只有16个计算线程逐行扫描64个分数，K/V加载同步，跨warp通过shared交换。这不是FA2完整调优的数据通路，当前数值也未匹配原Torch逐位。

新增流水只覆盖GEMM内部K块：先载入块0，再同时“加载块1/算块0”，交接后“加载块2/算块1”，循环至末尾。warp0领取的是CTA任务，warp1/2负责加载，warp3–6计算，warp7写回；不是每个warp都从独立ready队列领取任意算子。目前还没有上游tile一完成就驱动下游tile的跨阶段流水。
