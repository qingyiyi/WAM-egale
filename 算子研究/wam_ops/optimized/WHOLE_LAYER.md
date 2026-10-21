# 简易版完整单层：一个 persistent kernel

当前唯一整层优化入口为[whole_layer.py](whole_layer.py) → [whole_layer.cu](csrc/whole_layer.cu)。Python绑定原参数与KV；C++分配workspace；一次cooperative launch完成完整单层。没有在kernel里再启动子kernel，也不合并30层。

## 按这个顺序读源码

1. `whole_layer.py`：输入和原模型一致，返回hidden或供外部核验的中间量。
2. `whole_layer.cu`的`whole_layer()`：依照模型顺序列出全部计算，五段中文注释给出数据流。
3. `begin_stage → claim_tile → finish_tile`：warp0给整个CTA领取任务；tile结束后复用shared，阶段结束用grid.sync。
4. `linear/norm/point/attention`：决定各阶段怎样切tile，再调用设备端计算。
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
|Norm/RMS|行宽3072；普通Norm保持原128线程归约；B14长cross-K RMS每CTA两个独立128线程组算两行|
|pointwise|每任务2048元素，8个warp共同执行；GELU与BF16逐token调制表每线程搬运四元素；其他公式保留原路径|

108常驻CTA，每CTA8warp/256线程。warp0领取CTA任务，warp3–6计算。小M的warp1/2搬A/W、warp7写回；大M的4个计算warp自行加载和MMA，随后8warp共同做epilogue。普通Norm由8warp搬入/写回、3–6归约；双行RMS两个4warp组各归约一行。FA2沿用4warp设备计算。B14只读cache的mask由8warp按原物理slot顺序压紧。

当前遵循用户“先对齐再调度”的要求，各阶段仍依次执行并全局同步。此前独立ready试验已经留档并撤回活动路径，不纳入当前版本。中间量仍按原BF16/FP32边界写入全局workspace，一个kernel不等于所有数据都留在shared。

## 本次简化

删除核心的重复shape/版本检查、tile访问审计计数和未启用的旧FFN-ready分支；保留边界mask、合法cache逻辑、线程同步和CUDA错误报告。只维护这条简易执行路径。

精度/状态/输入/RNG检查在外部实验脚本；核心不再支持`audit`、`tile_pipeline`、`bf16_split_reduction`开关。原BF16分段归约固定保留，不把它当可关闭的检查。

历史复杂版源码和失败证据在[本轮证据目录](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/)，不作为活动优化入口；baseline、投机侧和原tile未改。

## 验证与性能边界

最新证据见[逐元素与FFN2剥离、整层结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/pointwise-isolation-20261021/RESULT.md)。
真实Torch GELU使用vec4；当前自有独立vec4约慢4%–6%，有限逐位相同。把table/GELU vec4放回整层，
两独立进程均改善约2.7%–3.1%，已保留。当前B14短/中/长完整层约为native的1.422/1.439/1.301–1.304倍，尚未速度对齐。

FFN2独立M128两K分段并行也接近Torch（慢约4%–6%），但集成整层未稳定受益，短KV反慢2%–4%，
已撤回并恢复本次较快vec4源/DSO。活动FFN2仍为M64串行两段；不要把证据目录的诊断版本当生产入口。
新增FFN2整层local从1464升到1488B只是静态观察，尚未证明动态spill或同步是唯一原因。
当前仍按阶段grid.sync，不启动跨阶段ready。保留夜间[RMS等原证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md)。

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
