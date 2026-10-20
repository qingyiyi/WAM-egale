# 当前完整单层：一个 persistent kernel

[whole_layer.py](whole_layer.py)绑定27组参数/7个eps、当前KV和workspace；[whole_layer.cu](csrc/whole_layer.cu)完成唯一cooperative launch。host不做GEMM/SDPA数学，kernel内部没有设备子kernel。范围只有一层，production_admitted=false。

## 数据、DAG与tile

输入 hidden 为BF16 `[B,Q,3072]`；context为BF16 `[B,512,3072]`；temb为原BF16/FP32调制张量；RoPE保留原complex64/FP64算术；KV保持原缓存和candidate_rows可见性。输出hidden形状/dtype不变，合法KV写回遵循原协议。

`Norm1/调制 → QKV/RMS/RoPE → self-FA2 → 投影/残差 → Norm2 → cross-QKV/RMS/FA2/投影/残差 → Norm3/调制 → FFN1/GELU → FFN2/门控残差`。

|计算|tile与分工|
|---|---|
|GEMM小M32|输出64×64，K32；四个compute warp各32×32|
|GEMM M≥128|输出64×128，K64；CUTLASS SM80 3级主循环，四个compute warp各32×64|
|FA2 self/cross|普通Q128/KV64；split Q64/KV128；head128，保留原softmax统计合并|
|Norm|整3072维行；向量IO，原Welford/LayerNorm设备体与归约顺序|
|pointwise|128元素任务，原BF16/FP32舍入边界|

108常驻CTA，每CTA8warp/256线程。warp0领CTA任务，3–6计算，7写回；小M的warp1/2加载A/W，大M CUTLASS由四个计算warp自行cp.async/load/MMA，warp1/2当前空闲；**不是每warp独立领取完整GEMM**。FA2当前仍由四个compute warp完成其原load/MMA/softmax/store，尚未改成专职load warp。

B14仍是14个独立Q16实例，不能写成Q112序列；共享history与private suffix直接按合法地址读取，不让候选互相注意。action短中长KV288/1152/2160，video272/1136/2144；当前B6/3候选与RoboTwin Q32未准入。

## 当前两条执行路线

默认 `tile_pipeline=False`：保留阶段grid barrier。

`tile_pipeline=True`：[ffn_pipeline.cuh](csrc/ffn_pipeline.cuh)将FFN1/GELU→FFN2改成完整64行组就绪交接。生产者保留FFN1与GELU的BF16边界；整组F14336列全部完成并release发布后，消费者acquire领取FFN2。保持完整K归约次序，不把一个列tile误作整个FFN2输入。所有CTA驻留，调度器先领取就绪consumer，否则领取producer，不阻塞等待未调度producer；bounded CAS避免无效ticket无限递增。

每调用重新初始化计数，workspace贯穿该调用，末尾grid barrier后回收。B14/video四行组实测存在跨阶段交接，单行组B2 action不具备该重叠；其他子图仍有阶段barrier。旧ready路径更慢，默认关闭；本次新CUTLASS二进制未单独重验ready，不能继承旧版通过。

## 数值与资源

[norm_codegen.py](norm_codegen.py)保留固定Torch LayerNorm设备计算；[gemm_pipeline.cuh](csrc/gemm_pipeline.cuh)按真实cuBLASLt记录保留有限形状的serial BF16归约、B2 sliced1x2及bias加法顺序。不能凭数学等价改变累加/舍入。

参数以 `const __grid_constant__ Params`传递。当前CUTLASS版108CTA、255 registers/thread、82464B shared、1536B localSizeBytes；不是实测occupancy/访存或spill流量。SM80不假设TMA/WGMMA或动态warp寄存器重新分配。

[当前CUTLASS结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)：默认路径9真实签名/每格23配对中间量/18有效KV转移逐位通过；输入/RNG/tile通过，重复门本轮未单独测。两个独立进程的B14短/中/长完整层中位7.9375/8.1437/8.6423ms，同轮native2.6537/2.9433/3.3851ms；仍慢2.54–3.02倍。六trace各1 compute kernel。旧版两路线9格、27trace和数值/速度负结果保留在[前版报告](../../../.pipeline/experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)，不混成新版ready证据。

[cutlass_gemm.cuh](csrc/cutlass_gemm.cuh)复用固定CUTLASS 3.5.1设备MmaMultistage；[生成器](cutlass_gemm_codegen.py)只改变类名及两个128-thread named barrier。没有调用host GEMM或设备子kernel；保留原BF16分段/bias顺序。这是成熟原语接入，不是新GEMM算法或已证明的MegaTransformer增量。

不包含30层/verifier/chunk/机器人E2E或最强cache/Graph，四项权限仍false，生产未准入。

## 构建与使用

固定Torch git `0fabc3ba44823f257e70ce397d989c8de5e362c1`、SM80、私有CUDA12.4，wheel为cu126；不启fast-math，不改baseline或vendor。额外构建依赖为固定CUTLASS commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`的完整include，位于项目`.pipeline/cache/cutlass-v3.5.1/include`；旧FA2裁剪vendor不包含完整GEMM头，不能替代。保留上游许可证，cache/DSO不推GitHub。设置 `WAM_FA2_BUILD_ROOT`为用户目录内构建路径，再执行 `python build_whole_layer.py build_ext --build-temp ... --build-lib ...`。不要在同一进程装载两份同名扩展进行A/B；这类初测已标INVALID并保留。

在 `torch.inference_mode()`内构造 `WholeLayerBlock(block)`。BF16 reduced-precision契约、固定设备/库及shape边界必须与报告一致；不支持autograd。debug返回中间量和metadata，只比较逻辑有效KV。旧M34/M39数值/资源FAILED、全部训练/闭环历史门保留，不能借本次有限通过追溯修改。
