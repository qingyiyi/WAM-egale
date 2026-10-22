# 预计算后完整阶段与耗时

适用：远端A100-SXM4-40GB、B14/Q16、D3072、FFN14336；两组文本条件各512token，原有self历史短/中/长KV288/1152/2160。正常版本一层一个persistent kernel；缓存由原Torch在GPU上预计算，warm主kernel直接读最终K/V。

下面是独立插桩版本的CTA0阶段时间，中位数来自8轮；包含阶段内计算、任务领取、CTA/warp同步及全grid等待。最后一列是同一CTA0落在该阶段内的grid等待，不是全GPU关键路径等待占比。屏障外时间也不能全部叫作纯计算。逐CTA分布及未缓存对照见CSV。阶段分量中位数之和不等于完整层中位数。

|序号|阶段|warm逻辑tile|每tile实际工作|短KV μs|中KV μs|长KV μs|中KV其中grid等待 μs|
|---|---|---:|---|---:|---:|---:|---:|
|1|LN1|224|1行×3072维：均值/方差归约、归一化|15.36|15.87|15.36|1.02|
|2|调制1|336|2048元素：按需scale/shift，N×(1+scale)+shift|9.22|9.22|9.22|1.02|
|3|Self Q投影|96|64×128输出，K=3072；tile内2段K1536|50.18|50.18|50.18|4.10|
|4|Self K投影|96|64×128输出，K=3072；tile内2段K1536|46.08|46.08|46.08|3.58|
|5|Self V投影|96|64×128输出，K=3072；tile内2段K1536|45.06|45.57|45.06|3.07|
|6|Self Q RMS+RoPE|112|2行×3072维RMS，再旋转3072个复数对|16.38|15.36|15.36|1.02|
|7|Self K RMS+RoPE|112|2行×3072维RMS，再旋转3072个复数对|14.34|14.34|14.34|1.02|
|8|Self attention|96/168/96|2个独立4-warp组；每组16/32个有效Q，沿KV每64行循环|60.42|232.96|247.81|10.24|
|9|Self输出投影|96|64×128输出，K=3072；tile内2段K1536|50.69|47.10|47.10|4.10|
|10|残差1|336|2048元素：X+gate1×SO，gate按需计算|10.24|10.24|10.24|2.05|
|11|LN2|224|1行×3072维：LN及affine|22.53|22.53|22.53|1.02|
|12|Cross Q投影|96|64×128输出，K=3072；tile内2段K1536|48.13|48.13|48.13|4.10|
|13|Cross K投影|0|warm无；未缓存自有路径128×128输出，K=3072|0.00|0.00|0.00|0.00|
|14|Cross V投影|0|warm无；未缓存自有路径128×128输出，K=3072|0.00|0.00|0.00|0.00|
|15|Cross Q RMS|112|2行×3072维：RMS，含缩放权重|11.26|10.24|10.24|1.02|
|16|Cross K RMS|0|warm无；未缓存自有路径2行×3072维RMS|0.00|0.00|0.00|0.00|
|17|Cross attention|168|2个独立4-warp组；每组Q16×K512，head_dim128，8轮KV64|74.75|74.75|74.75|2.56|
|18|Cross输出投影|96|64×128输出，K=3072；tile内2段K1536|49.15|48.64|48.13|4.61|
|19|残差2|336|2048元素：R1+CO|8.19|9.22|9.22|2.05|
|20|LN3|224|1行×3072维：均值/方差归约、归一化|23.55|23.04|23.04|2.05|
|21|调制3|336|2048元素：按需scale/shift，N×(1+scale)+shift|10.24|11.26|10.24|2.05|
|22|FFN1|112|128×256输出，K=3072；96轮K32|199.68|199.17|198.14|3.07|
|23|GELU|1568|2048元素：GELU_tanh|30.21|30.72|31.74|2.56|
|24|FFN2|96分段任务/48输出tile|128×128部分和，K=7168；112轮K64；两任务合成一个输出tile|139.26|139.26|139.26|15.36|
|25|残差3|336|2048元素：R2+gate3×F2，gate按需计算|11.26|10.75|11.26|2.56|
|26|layout|—|实际在最前：CTA0每轮扫描256个物理槽，压紧合法history|9.22|9.22|9.22|1.02|
|27|exit|—|全grid退出屏障，无数学tile|1.54|2.05|2.05|1.02|

## tile口径与每个tile的工作量（2026-10-22补充）

上表新增工作量来自当前源码静态核查，时间仍来自原独立profile，没有新启动GPU、重新计时或更改计算代码。适用本次B14真实输入：M=14×16=224，D=3072，FFN=14336。单kernel启动108个常驻CTA、每CTA256线程/8warp；表中tile是阶段逻辑工作包数，不是新kernel数量，也不是新启动CTA数。

- LN：224行，每tile独立归约完整3072维。RMS：两组4warp各处理一行，每tile两行，224/2=112；RMS+RoPE随后处理两行共3072个复数对，保留BF16中间舍入和原FP64旋转。
- 调制/残差：224×3072/2048=336；GELU：224×14336/2048=1568。一次向量搬4个元素，每CTA的256线程各处理8个有效标量。它们是向量化访存和逐元素公式，不能按GEMM tile衡量工作量。
- 六个action投影：ceil(224/64)×(3072/128)=96个输出tile。72个tile为64个有效行，24个尾tile只有32个有效行。每tile内部顺序完成两个K1536段（每段24轮K64），保留中间BF16提交；这些分段没有变成192个队列任务。
- FFN1：ceil(224/128)×(14336/256)=112个输出tile。56个tile为128有效行，另56个尾tile为96有效行；8warp做MMA。静态映射在108个CTA上，CTA0–3各多处理一个tile。
- FFN2：ceil(224/128)×(3072/128)=48个输出tile；每个拆成两个K7168部分和，故96个计算任务。24个输出tile为128有效行、24个为96有效行。前96个CTA各执行一个部分和，余12个没有MMA任务；第一段BF16提交后grid同步，第二段再加bias和前段并写回，最后grid同步。不能把96解释为96个互不相同的输出tile。
- 预计算后Cross K投影/V投影/K RMS在warm中均为0。未缓存但已分组的自有路径原为192/192/512，移出共896任务。预计算由Torch在GPU执行；这三个旧自有计数不是Torch冷构建的CTA数，冷构建实际kernel tile未在本轮统计。

### GEMM工作量：满tile与有效尾行分别计算

下表仅为矩阵乘加的理论工作量，1次乘加按2 FLOPs；不含bias、归约、舍入、访存、同步，也不是profiler实测执行FLOPs。尾行的有效数学工作减少，不能据此认为硬件对padding的MMA指令必然省掉。

|类型|一个任务的M×N×K|满tile乘加工作量|尾tile有效工作量|
|---|---|---:|---:|
|六个action投影|64×128×3072|50.332 MFLOPs|32行：25.166 MFLOPs|
|未缓存分组Cross K/V|128×128×3072|100.663 MFLOPs|本次512token整除，无尾行|
|FFN1|128×256×3072|201.327 MFLOPs|96行：150.995 MFLOPs|
|FFN2的一个K分段|128×128×7168|234.881 MFLOPs|96行：176.161 MFLOPs|

### attention：外层工作包与内部FA2循环分开

每个表中attention工作包打包两个独立4-warp组。每组内部FA2 tile为Q64×KV64、head_dim128；有效Q只有16或32。QK和PV各执行矩阵乘，期间有online softmax。每个query只看公共prefix与自己的16个private token，跨候选private位置被mask；没有新增可任意互相注意的Q112序列。

|KV档|合法prefix|每query可见KV（含自身16）|Self分组|4-warp组任务数|表中CTA工作包数|每组KV64循环|
|---|---:|---:|---|---:|---:|---:|
|短|272|288|G2|192|96|5|
|中|1136|1152|G1|336|168|18|
|长|2144|2160|G2|192|96|34|
|Cross三档相同|不适用|512个context token|逐candidate/head|336|168|8|

G2按每CFG的7候选分成2/2/2/1；192个4-warp任务中144个处理Q32、48个处理Q16。按当前源码相邻两任务打包，96个CTA工作包中48个处理32+32个有效Q、48个处理32+16个有效Q。两组独立计算，并不合成一个可跨组注意的矩阵。G1每工作包为16+16个有效Q。中KV若强行G2会让内部长度从1152变1168，KV64循环从18变19，因此当前guard选G1。

只统计合法QK/PV的有效乘加，每4-warp组为4×q_valid×L_visible×128 FLOPs（另加softmax等）；一个工作包对两组求和。G2内部仍计算部分被mask的private位置和padding，不能把此有效公式当硬件执行FLOPs；固定64×64×128主循环的两次GEMM理论为2.097152 MFLOPs/组/轮，与有效数学工作量分列。

### 同步与源码定位

warm正常路径31次grid同步：六个动态投影各入口/出口两次（12）；FFN2两次；其他15个计算阶段各一次（15）；layout入口准备完成及最终exit各一次（2），总计12+2+15+2=31。此处阶段之间仍同步，没有恢复跨阶段ready调度。CTA内barrier与FA2组内barrier另计，不包含在31中。

源码依据（相对项目根目录）：
- `算子研究/wam_ops/optimized/csrc/whole_layer.cu:25`：D/H/head_dim/F/线程数与FA2 tile；`:145`六投影；`:178`context tile；`:247`FFN2；`:265`FFN1；`:490`norm；`:512`RMS+RoPE；`:521`layout；`:633`point tile；`:752`实际阶段调用顺序。
- `算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh:15`：GEMM threadblock/warp/instruction形状；`:64`K分段、BF16舍入与写回。
- `算子研究/wam_ops/optimized/csrc/fa2_stage.cuh:165`：双组attention、G1/G2判定、任务计数及候选可见范围。
- `算子研究/wam_ops/optimized/build_whole_layer.py:17`：本次context tile行数构建默认128（头文件fallback64不能冒充实测配置）。
- 原始计时：`STAGE-TIMING.csv`；原真实prefix：`STRUCTURAL-RESULT.json`各档`original`行的`valid_history`；当前版本身份：`SOURCE-MANIFEST.json`、`DEPLOYMENT-LOCAL.json`。

## 计算公式

|序号|计算|
|---|---|
|1|LN(X)|
|2|N1*(1+scale1)+shift1|
|3|M1 Wqᵀ+bq|
|4|M1 Wkᵀ+bk|
|5|M1 Wvᵀ+bv|
|6|RoPE(BF16(RMS(Q0)))|
|7|RoPE(BF16(RMS(K0)))|
|8|softmax(QrKvisibleᵀ/√128)Vvisible|
|9|SA Woᵀ+bo|
|10|X+gate1*SO|
|11|affine_LN(R1)|
|12|N2 Wcqᵀ+bcq|
|13|预计算，warm跳过|
|14|预计算，warm跳过|
|15|RMS(Q0)|
|16|已含在预计算K中|
|17|softmax(QKgroupᵀ/√128)Vgroup|
|18|CA Wcoᵀ+bco|
|19|R1+CO|
|20|LN(R2)|
|21|N3*(1+scale3)+shift3|
|22|M3 W1ᵀ+b1|
|23|GELU_tanh(F1)|
|24|A W2ᵀ+b2，原两段舍入|
|25|R2+gate3*F2|
|26|有效history slot布局|
|27|退出同步|

普通warm路径有22个计算阶段、G2/G1为5304/5376逻辑tile、31次grid同步；uncached分组路径仍6200/6272tile、36次。预计算移出两投影和K RMS，共896逻辑tile、5次grid同步。layout/exit单列；debug额外物化调制表。

Cross K0/K/V分别保留两组，每张6MiB，共18MiB；其中原始K0额外保留供审计，warm计算实际只读最终K/V共12MiB。guarded保持原接口，预先展开B14缓存，三张总126MiB；展开在cold阶段计入，warm不展开。不是容量最优生产缓存。

## 插桩扰动

|KV|模式|正常完整层 ms|插桩Graph ms|扰动|
|---|---|---:|---:|---:|
|short|grouped_uncached|1.208832|1.230192|+1.77%|
|short|cached|0.961520|0.980320|+1.96%|
|middle|grouped_uncached|1.374480|1.396144|+1.58%|
|middle|cached|1.128448|1.149472|+1.86%|
|long|grouped_uncached|1.392128|1.415536|+1.68%|
|long|cached|1.142784|1.162160|+1.70%|

插桩额外写全局时间戳，并清零诊断buffer，因此上表不能直接作为正常路径的绝对阶段耗时；正常整层速度单独测量。所有两模式×三KV的profile输出/共有中间量/有效KV/缓存/输入/RNG均有限核验通过。两处同步漏计的原失败保留在首PROFILE-RESULT，修复版见profile-r2。

[正常完整层](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/WHOLE-LAYER-TIMING.csv)、[逐阶段CSV](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/STAGE-TIMING.csv)、[诊断汇总](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/PROFILE-SUMMARY.json)、[原始诊断](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/profile-r2/PROFILE-RESULT.json)、[结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)。
