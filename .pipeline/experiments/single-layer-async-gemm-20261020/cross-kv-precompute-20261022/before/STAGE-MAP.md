# 当前Cross K/V分组复用版：完整阶段表

有限验证范围：B14 action、hidden=[14,16,3072]，原context=[14,512,3072]，按字节相同条件准备为两组。Q总行224，Cross K/V各1024行（原7168），D=3072，FFN=14336。每组512个文本token。108个persistent CTA，每CTA256线程/8warp，一层一个计算kernel。

下表序号为阅读顺序，不是旧profiler阶段编号。tile为逻辑工作块，不是CTA数或原子领取次数；大部分静态block-stride，八投影阶段动态队列。实数公式仅解释依赖，BF16/FP32/FP64边界按代码保留。

|顺序|阶段|计算|tile形状|逻辑tile数|全grid同步|
|---|---|---|---|---:|---|
|1|LN1|N1=LN(X)|1行3072维|224|后1次|
|2|调制1|M1=N1*(1+scale1)+shift1|2048元素|336|后1次|
|3|self Q投影|Q0=M1 Wqᵀ+bq|64×128|96|前1、后1次|
|4|self K投影|K0=M1 Wkᵀ+bk|64×128|96|前1、后1次|
|5|self V投影|V=M1 Wvᵀ+bv|64×128|96|前1、后1次|
|6|self Q RMS+RoPE|Qr=RoPE(BF16(RMS(Q0)))|两行|112|后1次|
|7|self K RMS+RoPE|Kr=RoPE(BF16(RMS(K0)))|两行|112|后1次|
|8|self attention|softmax(Qr Kvisibleᵀ/√128)Vvisible|两个4warp子任务|G2:96 / G1:168|后1次|
|9|self输出投影|SO=SA Woᵀ+bo|64×128|96|前1、后1次|
|10|残差1|R1=X+gate1*SO|2048元素|336|后1次|
|11|LN2|N2=affine_LN(R1)|1行|224|后1次|
|12|cross Q投影|Q0=N2 Wcqᵀ+bcq|64×128|96|前1、后1次|
|13|cross K投影|K0=context_group Wckᵀ+bck|每组128×128|192|前1、后1次|
|14|cross V投影|V=context_group Wcvᵀ+bcv|每组128×128|192|前1、后1次|
|15|cross Q RMS|Q=RMS(Q0)|两行|112|后1次|
|16|cross K RMS|K=RMS(K0)|两行|512|后1次|
|17|cross attention|softmax(Q Kgroupᵀ/√128)Vgroup|两个独立4warp子任务|168|后1次|
|18|cross输出投影|CO=CA Wcoᵀ+bco|64×128|96|前1、后1次|
|19|残差2|R2=R1+CO|2048元素|336|后1次|
|20|LN3|N3=LN(R2)|1行|224|后1次|
|21|调制3|M3=N3*(1+scale3)+shift3|2048元素|336|后1次|
|22|FFN1|F1=M3 W1ᵀ+b1|128×256|112|后1次|
|23|GELU|A=GELU_tanh(F1)|2048元素|1568|后1次|
|24|FFN2|F2=A W2ᵀ+b2，保持原两段舍入|48输出tile×两K7168段|96|两段各1次|
|25|残差3|Y=R2+gate3*F2|2048元素|336|后1次|

普通路径合计G2=6200、G1=6272；短/中/长fixture分别用G2/G1/G2。相比前版少5376逻辑tile：两Cross投影各少1152，Cross K RMS少3072。Q及每个候选自己的注意力边界不变；B14仍为14个Q16实例。Cross复用范围按context判断，不能按history映射猜测。

全grid同步仍36：layout1+25阶段出口+8投影队列入口+FFN2中间1+退出1。B14不将候选private K/V写入公共history。attention内部额外组同步、cp.async等待仍保留。debug物化MOD并展开Cross K/V以便审计，为37次grid；普通计时无此展开。

全层仍每次计算两组Cross K/V，不是跨调用预计算缓存。`WholeLayerBlock.prepare_context(context)`在Graph外调用，context变化后必须重做；未调用时走完整形状兼容路径，不能套用本表的去重tile数。B2保持原兼容路径。

FFN2按原两个K段执行：先提交第一段BF16，第二段保留FP32局部结果，grid同步后按原顺序与bias、第一段结果合并；不随意改为两段FP32直接相加。RMS仍需整行完整归约，本轮没有与GEMM tile融合。

资源108CTA/100096B shared/255regs/static local936B，来自运行时资源字段；不是新动态occupancy或spill profile。Cross三中间buffer总分配126→18MiB，不是实测HBM/peak下降。

[结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-reuse-20261022/RESULT.md)；源码入口`csrc/whole_layer.cu:whole_layer/context_linear`、`csrc/fa2_stage.cuh:attention_dual`、`whole_layer.py:prepare_context`。
