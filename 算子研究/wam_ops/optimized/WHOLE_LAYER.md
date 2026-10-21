# 简易版完整单层：一个 persistent kernel

当前唯一整层优化入口为[whole_layer.py](whole_layer.py) → [whole_layer.cu](csrc/whole_layer.cu)。Python绑定原参数与KV；C++分配workspace；一次cooperative launch完成完整单层。没有在kernel里再启动子kernel，也不合并30层。

## 按这个顺序读源码

1. `whole_layer.py`：输入和原模型一致，返回hidden或供外部核验的中间量。
2. `whole_layer.cu`的`whole_layer()`：依照模型顺序列出全部计算，五段中文注释给出数据流。
3. `begin_stage → claim_tile → finish_tile`：warp0给整个CTA领取任务；tile结束后复用shared，阶段结束用grid.sync。
4. `linear/context_linear/action_ffn_linear/action_ffn2_parallel/norm/point/attention`：决定各阶段怎样切tile，再调用设备端计算。
5. [cutlass_gemm.cuh](csrc/cutlass_gemm.cuh)、[gemm_pipeline.cuh](csrc/gemm_pipeline.cuh)、[fa2_stage.cuh](csrc/fa2_stage.cuh)：GEMM、内部FA2和合法KV寻址。

## 输入与计算

hidden：BF16 `[B,Q,3072]`；context：BF16 `[B,512,3072]`；temb：原BF16/FP32调制张量；RoPE保留原算术。输出hidden同形状、同dtype。B14对应14个独立Q16实例，候选private KV互不可见；不能解释为Q112序列。

`Norm1/调制 → self QKV/RMS/RoPE/FA2 → 投影/残差 → Norm2 → cross QKV/RMS/FA2 → 投影/残差 → Norm3/调制 → FFN1/GELU/FFN2 → 门控残差`。

|阶段|任务划分|
|---|---|
|B2 action，小M32 GEMM|64×64输出tile，K32，独立加载warp与WMMA|
|B14 action，M224 GEMM|投影64×128；FFN1/FFN2为128×128；K64，CUTLASS三级主循环；FFN2另见下文|
|context，M7168 GEMM|128×128输出tile，K64；单次K/V投影各1344任务|
|self/cross FA2|B14 Q16实例用Q64×KV64 tile；其余standard为Q128×KV64，需要split时Q64×KV128；连续cross K/V保留原FA2加载器|
|Norm/RMS|行宽3072；普通Norm保持原128线程归约；candidate RMS每CTA两个独立128线程组直接读取并各算一行，224行自然形成112任务|
|pointwise|每任务2048个逻辑单位，8个warp共同执行；GELU、BF16逐token调制表、调制与残差每线程搬运4元素；RoPE每线程处理2个复数对并保留FP64算术，RoPE任务单位为复数对|

108常驻CTA，每CTA8warp/256线程。warp0领取CTA任务，warp3–6计算。小M的warp1/2搬A/W、warp7写回；大M的4个计算warp自行加载和MMA，随后8warp共同做epilogue。普通Norm由8warp搬入/写回、3–6归约；双行RMS两个4warp组各归约一行。FA2沿用4warp设备计算。B14只读cache的mask由8warp按原物理slot顺序压紧。

每类子算子按适合的tile大小与逻辑任务数划分，108是当前常驻worker池规模，不是每阶段必须凑满108任务。例如Self Q为96个输出tile；FFN2为48个输出tile×2个K分段=96个计算任务，candidate self RMS为112个双行任务。未来从真实DAG中把已就绪的后续tile分配给空闲worker；不会为铺满而强行切碎当前算子。

后续依赖设计先区分：cross K/V只读就绪context和各自权重，可提前；Self Q的RMS需对应行全部3072列投影完成，不能单个输出tile完成就运行；FFN2需其完整reduction输入就绪，不能看到一个GELU tile就提前。任务不就绪时不能让全部常驻CTA占着资源自旋，阻塞真正的生产者。这些是下一步调度约束，当前未实现。

当前遵循用户“先对齐再调度”的要求，各阶段仍依次执行并全局同步。此前独立ready试验已经留档并撤回活动路径，不纳入当前版本。中间量仍按原BF16/FP32边界写入全局workspace，一个kernel不等于所有数据都留在shared。

B14 FFN2在`action_ffn2_parallel`内将两个K7168分段分配给两组CTA同时计算。第0段先写BF16结果，经grid.sync后第1段按`(FP32+bias)+首段BF16`顺序归并，再写BF16；保持原舍入顺序。整个阶段是一个专用设备函数，partition/commit在内部展开。仍有阶段和归并同步，没有跨阶段ready。

## 本次简化

删除核心的重复shape/版本检查、tile访问审计计数和未启用的旧FFN-ready分支；保留边界mask、合法cache逻辑、线程同步和CUDA错误报告。只维护这条简易执行路径。

精度/状态/输入/RNG检查在外部实验脚本；核心不再支持`audit`、`tile_pipeline`、`bf16_split_reduction`开关。原BF16分段归约固定保留，不把它当可关闭的检查。

历史复杂版源码和失败证据在[本轮证据目录](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/)，不作为活动优化入口；baseline、投机侧和原tile未改。

## 验证与性能边界

当前证据：[FFN资源控制与完整层](../../../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md)、[整层退化原因](../../../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/CAUSE-ANALYSIS.md)。

在前轮Cross/FFN函数专门化之后，本轮保留专用FFN2分段并行的小幅收益：两进程三KV完整层下降0.36%–0.65%；短/中/长约2.745/3.007/3.262ms，仍比native慢29.6%–29.8% / 25.1%–25.2% / 14.7%–14.9%。只指uncached/eager完整单层，不是强cached/Graph基线、30层、chunk或E2E。

九个真实签名（B2 action/B14 action/B2 video×三KV）、每格23中间量、18有效KV转移、输入/RNG和重复输出有限逐位通过，max_abs=0。一个kernel；不外推任意输入、闭环质量或正式生产准入。

同源码、相同前序、同设备函数调用点计时，独立与整层差小于约1.4%；旧的大额集成惩罚已明显缓解。当前FFN1/FFN2仍比native原语慢，先改进原语再推进DAG。静态寄存器255、shared98848B、1CTA/SM不等于动态occupancy/spill已测；后者UNKNOWN。

历史记录保留：

- [Cross/FFN函数组织控制](../../../.pipeline/experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md)：固定tile/数学/sync，前后完整层下降9.46%–10.87%；不与本轮倍数相乘。
- [小算子与CTA等待](../../../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md)：point向量化、candidate RMS双行；旧29阶段表属于当时源。
- [pointwise与旧FFN2负结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md)：旧generic linear内并行K独立快但整层负收益；记录不改，当前使用专用阶段组织，不声称两个版本只差单一因素。
- [简化及阶段对齐](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)：保留复杂版和负结果，仅当前简易版为活动代码。

N256/8warp FFN1、pointwise8192、强制通用GEMM内联、双驻留compact B14的旧负结果均保留。当前仍阶段grid.sync，跨阶段ready未实现。早到CTA等待与其他CTA计算重叠，不等于纯barrier耗时或全部可兑现收益。
