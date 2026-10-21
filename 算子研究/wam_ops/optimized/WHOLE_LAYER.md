# 简易版完整单层：一个 persistent kernel

当前唯一整层优化入口为[whole_layer.py](whole_layer.py) → [whole_layer.cu](csrc/whole_layer.cu)。Python绑定原参数与KV；C++分配workspace；一次cooperative launch完成完整单层。没有在kernel里再启动子kernel，也不合并30层。

## 按这个顺序读源码

1. `whole_layer.py`：输入和原模型一致，返回hidden或供外部核验的中间量。
2. `whole_layer.cu`的`whole_layer()`：依照模型顺序列出全部计算，五段中文注释给出数据流。
3. `begin_stage → claim_tile → finish_tile`：warp0给整个CTA领取任务；tile结束后复用shared，阶段结束用grid.sync。
4. `linear/context_linear/action_ffn_linear/norm/point/attention`：决定各阶段怎样切tile，再调用设备端计算。
5. [cutlass_gemm.cuh](csrc/cutlass_gemm.cuh)、[gemm_pipeline.cuh](csrc/gemm_pipeline.cuh)、[fa2_stage.cuh](csrc/fa2_stage.cuh)：GEMM、内部FA2和合法KV寻址。

## 输入与计算

hidden：BF16 `[B,Q,3072]`；context：BF16 `[B,512,3072]`；temb：原BF16/FP32调制张量；RoPE保留原算术。输出hidden同形状、同dtype。B14对应14个独立Q16实例，候选private KV互不可见；不能解释为Q112序列。

`Norm1/调制 → self QKV/RMS/RoPE/FA2 → 投影/残差 → Norm2 → cross QKV/RMS/FA2 → 投影/残差 → Norm3/调制 → FFN1/GELU/FFN2 → 门控残差`。

|阶段|任务划分|
|---|---|
|B2 action，小M32 GEMM|64×64输出tile，K32，独立加载warp与WMMA|
|B14 action，M224 GEMM|投影/FFN2为64×128；FFN1为128×128；K64，CUTLASS三级主循环|
|context，M7168 GEMM|128×128输出tile，K64；单次K/V投影各1344任务|
|self/cross FA2|B14 Q16实例用Q64×KV64 tile；其余standard为Q128×KV64，需要split时Q64×KV128；连续cross K/V保留原FA2加载器|
|Norm/RMS|行宽3072；普通Norm保持原128线程归约；candidate RMS每CTA两个独立128线程组直接读取并各算一行，224行自然形成112任务|
|pointwise|每任务2048个逻辑单位，8个warp共同执行；GELU、BF16逐token调制表、调制与残差每线程搬运4元素；RoPE每线程处理2个复数对并保留FP64算术，RoPE任务单位为复数对|

108常驻CTA，每CTA8warp/256线程。warp0领取CTA任务，warp3–6计算。小M的warp1/2搬A/W、warp7写回；大M的4个计算warp自行加载和MMA，随后8warp共同做epilogue。普通Norm由8warp搬入/写回、3–6归约；双行RMS两个4warp组各归约一行。FA2沿用4warp设备计算。B14只读cache的mask由8warp按原物理slot顺序压紧。

每类子算子按适合的tile大小与逻辑任务数划分，108是当前常驻worker池规模，不是每阶段必须凑满108任务。例如Self Q和FFN2各96个GEMM任务，candidate self RMS为112个双行任务。未来从真实DAG中把已就绪的后续tile分配给空闲worker；不会为铺满而强行切碎当前算子。

后续依赖设计先区分：cross K/V只读就绪context和各自权重，可提前；Self Q的RMS需对应行全部3072列投影完成，不能单个输出tile完成就运行；FFN2需其完整reduction输入就绪，不能看到一个GELU tile就提前。任务不就绪时不能让全部常驻CTA占着资源自旋，阻塞真正的生产者。这些是下一步调度约束，当前未实现。

当前遵循用户“先对齐再调度”的要求，各阶段仍依次执行并全局同步。此前独立ready试验已经留档并撤回活动路径，不纳入当前版本。中间量仍按原BF16/FP32边界写入全局workspace，一个kernel不等于所有数据都留在shared。

## 本次简化

删除核心的重复shape/版本检查、tile访问审计计数和未启用的旧FFN-ready分支；保留边界mask、合法cache逻辑、线程同步和CUDA错误报告。只维护这条简易执行路径。

精度/状态/输入/RNG检查在外部实验脚本；核心不再支持`audit`、`tile_pipeline`、`bf16_split_reduction`开关。原BF16分段归约固定保留，不把它当可关闭的检查。

历史复杂版源码和失败证据在[本轮证据目录](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/)，不作为活动优化入口；baseline、投机侧和原tile未改。

## 验证与性能边界

当前最新是[Cross K/V与FFN函数组织控制](../../../.pipeline/experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md)。
固定tile/数学/同步，Cross使用整个阶段专用设备函数，FFN仅B14固定形状；tile循环内展开CUTLASS核心，减少反复进入通用设备函数的成本。
两独立进程直接测本轮前/最终版，短中长KV完整层下降10.84%–10.87% / 9.99%–10.13% / 9.46%–9.59%。
最终约2.766/3.028/3.281ms，仍比native慢30.4%–30.5% / 25.8%–25.9% / 15.3%–15.4%。
两阶段各9真实签名、23中间量、18有效KV状态及输入/RNG有限逐位通过；仍一个kernel。函数组织改变不等于新的ready调度，静态寄存器仍255，动态spill/traffic仍未知。
FFN单独增量约0.6%，尚未性能对齐；下一步先FFN原语/资源差距。本次没有恢复此前负收益FFN2并行分段版本。

以下为前轮历史，对应原件保留，不代表上述最新源：

最新证据见[RMS/RoPE/调制/残差、完整阶段及CTA等待](../../../.pipeline/experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md)。
本轮12个小算子三KV/两进程独立数学体有限逐位通过；RMS接近Torch，融合RoPE、调制和门控残差更快，直接残差仍略慢。
原Torch表达式可能启动多个kernel，旧“独立阶段”仍带统一执行器/shared预留，二者不应混称。
保留point向量化和candidate RMS双行直接读取后，固定60轮预热的两个独立进程，整层较本轮前改善2.2%–3.0%。
该前轮B14短/中/长约3.10/3.37/3.62ms，为native的1.463–1.467/1.402–1.403/1.274–1.275倍，尚未速度对齐。
这是uncached/eager单层；固定预热前的所有轮次仍保留，不能跨轮拼比例。此前table/GELU vec4改善2.7%–3.1%见[前轮结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md)，不与本轮改善直接相乘。

FFN2独立M128两K分段并行也接近Torch（慢约4%–6%），但集成整层未稳定受益，短KV反慢2%–4%，
已撤回并恢复本次较快vec4源/DSO。活动FFN2仍为M64串行两段；不要把证据目录的诊断版本当生产入口。
新增FFN2整层local从1464升到1488B只是静态观察，尚未证明动态spill或同步是唯一原因。
当前仍按阶段grid.sync，不启动跨阶段ready。新CTA时间戳副本中，中KV Cross K/V结束处等待P50约53µs，最晚进入到释放约2–3µs；早到等待与其他CTA计算重叠，不能称为纯同步开销或全部可兑现的收益。新的全29阶段表已重测；保留夜间[RMS等原证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md)。

N256/8warp FFN1、pointwise8192、强制GEMM内联、双驻留compact B14均未保留。后者虽达到2CTA/SM，完整层反而慢2%–10%；已恢复当前较快顺序版，当前FFN1仍N128。不把理论并行度或原语局部收益当作完整层收益。

此前结果见[简化与阶段对齐报告](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)。当前简易版已完成9种真实输入：B2 action、B14 action、B2 video，各短/中/长KV。本轮vec4与撤回的FFN2整层版本均核验最终输出、每格23个可配对中间量、18个有效KV状态转移，以及输入/RNG/重复结果有限逐位通过。完整单层新旧版计时与阶段测量以该报告终态为准。

本研究只覆盖固定SM80/Torch与有限真实输入；不是任意shape证明、生产准入、30层、完整chunk或机器人E2E。四项科学/正式性能权限仍false。NCU硬件counter权限未开放，动态occupancy/spill/HBM/L2未知。

## 构建与调用

固定Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`，CUTLASS `f7b19de32c5d1f3cedfc735c2849f12b537522ee`；私有CUDA12.4、Torch cu126、SM80，不用fast-math。`fa2_codegen.py`、`norm_codegen.py`、`cutlass_gemm_codegen.py`生成内部设备体；保留上游许可证。头文件有顺序依赖，格式化时禁自动排序include。

设置`WAM_FA2_BUILD_ROOT`为项目内构建目录，执行`build_whole_layer.py build_ext --build-temp ... --build-lib ...`。当前context tile默认128。运行时：

```python
from wam_ops.optimized.whole_layer import WholeLayerBlock
with torch.inference_mode():
    layer = WholeLayerBlock(block)
    output = layer(*args, **kwargs)
```

`debug=True`暴露现有中间量用于外部验证；不改变数学路径。对比不同版本须隔离模块名和kernel命名，不能误加载同名旧DSO。
