# 预计算后完整阶段与耗时

适用：远端A100-SXM4-40GB、B14/Q16、D3072、FFN14336；两组文本条件各512token，原有self历史短/中/长KV288/1152/2160。正常版本一层一个persistent kernel；缓存由原Torch在GPU上预计算，warm主kernel直接读最终K/V。

下面是独立插桩版本的CTA0阶段时间，中位数来自8轮；包含阶段内计算、任务领取、CTA/warp同步及全grid等待。最后一列是同一CTA0落在该阶段内的grid等待，不是全GPU关键路径等待占比。屏障外时间也不能全部叫作纯计算。逐CTA分布及未缓存对照见CSV。阶段分量中位数之和不等于完整层中位数。

|序号|阶段|warm逻辑tile|短KV μs|中KV μs|长KV μs|中KV其中grid等待 μs|
|---|---|---:|---:|---:|---:|---:|
|1|LN1|224|15.36|15.87|15.36|1.02|
|2|调制1|336|9.22|9.22|9.22|1.02|
|3|Self Q投影|96|50.18|50.18|50.18|4.10|
|4|Self K投影|96|46.08|46.08|46.08|3.58|
|5|Self V投影|96|45.06|45.57|45.06|3.07|
|6|Self Q RMS+RoPE|112|16.38|15.36|15.36|1.02|
|7|Self K RMS+RoPE|112|14.34|14.34|14.34|1.02|
|8|Self attention|96/168|60.42|232.96|247.81|10.24|
|9|Self输出投影|96|50.69|47.10|47.10|4.10|
|10|残差1|336|10.24|10.24|10.24|2.05|
|11|LN2|224|22.53|22.53|22.53|1.02|
|12|Cross Q投影|96|48.13|48.13|48.13|4.10|
|13|Cross K投影|0|0.00|0.00|0.00|0.00|
|14|Cross V投影|0|0.00|0.00|0.00|0.00|
|15|Cross Q RMS|112|11.26|10.24|10.24|1.02|
|16|Cross K RMS|0|0.00|0.00|0.00|0.00|
|17|Cross attention|168|74.75|74.75|74.75|2.56|
|18|Cross输出投影|96|49.15|48.64|48.13|4.61|
|19|残差2|336|8.19|9.22|9.22|2.05|
|20|LN3|224|23.55|23.04|23.04|2.05|
|21|调制3|336|10.24|11.26|10.24|2.05|
|22|FFN1|112|199.68|199.17|198.14|3.07|
|23|GELU|1568|30.21|30.72|31.74|2.56|
|24|FFN2|96|139.26|139.26|139.26|15.36|
|25|残差3|336|11.26|10.75|11.26|2.56|
|26|layout|—|9.22|9.22|9.22|1.02|
|27|exit|—|1.54|2.05|2.05|1.02|

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
