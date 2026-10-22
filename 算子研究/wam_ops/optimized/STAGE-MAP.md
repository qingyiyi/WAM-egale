# 当前按需调制版：从输入到输出

范围：在旧11上仅融合调制参数的已验证有限版本，B14 action、hidden=[14,16,3072]、context=[14,512,3072]，有RoPE和affine Norm2、uncached context。M=224，Mc=7168，D=3072，F=14336；108个常驻CTA，每CTA256线程/8warp，一个完整层计算kernel。

`P_W(X)=X W^T+b`；`LN(X)=(X-mean)/sqrt(var+eps)`，Norm2另有gamma/beta；`RMS(X)=gamma*X/sqrt(mean(X²)+eps)`。下表是数学简写，真实BF16/FP32/FP64与分段舍入按源码保留，不能据公式随意重排。

“tile数”是每阶段覆盖工作量的逻辑分块数，不是同时启动CTA数，也不都是原子领取次数。多数point/norm/FFN/attention静态block-stride；只有8个投影阶段使用动态队列。GEMM K方向内部MMA循环不另算调度任务。

|顺序|阶段/公式|一个tile及逻辑tile数|全grid同步|
|---|---|---|---|
|准备|整理有效history slot/layout|专用初始化，不纳入下面tile合计|后1次|
|原1移除|普通路径不物化MOD；在3/13/24/28处按需相加|0独立tile|无|
|2|N1=LN(X)|1行3072维；224|后1次|
|3|M1=N1*(1+scale1)+shift1；本tile计算scale/shift|2048元素；336|后1次|
|4|self Q0=Pq(M1)|64×128；ceil(224/64)×24=96|队列清零后1次、阶段后1次|
|5|self K0=Pk(M1)|同上96|前后各1次|
|6|self V=Pv(M1)|同上96|前后各1次|
|7|self Qn=RMS(Q0)|两行/CTA tile；112|后1次|
|8|self Kn=RMS(K0)|两行；112|后1次|
|9|self Qr=RoPE(Qn)|2048复数对=4096标量；168|后1次|
|10|self Kr=RoPE(Kn)|同上168|后1次|
|缓存入口|B14 private KV不写公共history|0任务，直接返回|0|
|11|SA=softmax(Qr Kvisible^T/sqrt(128))Vvisible|两组4warp attention子任务/CTA tile；G2为96、G1为168|后1次|
|12|SO=Po(SA)|64×128；96|前后各1次|
|13|R1=X+gate1*SO；本tile计算gate1|2048元素；336|后1次|
|14|N2=affine_LN(R1)|1行；224|后1次|
|15|cross Q0=Pcq(N2)|64×128；96|前后各1次|
|16|cross K0=Pck(context)|128×128；7168/128×24=1344|前后各1次|
|17|cross V=Pcv(context)|128×128；1344|前后各1次|
|18|cross Q=RMS(Q0)|两行；112|后1次|
|19|cross K=RMS(K0)|两行；7168/2=3584|后1次|
|20|CA=softmax(Q K^T/sqrt(128))V|14×24个head实例，2组/CTA tile；168|后1次|
|21|CO=Pco(CA)|64×128；96|前后各1次|
|22|R2=R1+CO|2048元素；336|后1次|
|23|N3=LN(R2)|1行；224|后1次|
|24|M3=N3*(1+scale3)+shift3；本tile计算scale/shift|2048元素；336|后1次|
|25|F1=P1(M3)，3072→14336|128×256；ceil(224/128)×56=112|后1次|
|26|A=GELU_tanh(F1)|2048元素；224×14336/2048=1568|后1次|
|27|F2=P2(A)，14336→3072|128×128输出48tile×两K7168段=96计算任务|第0段BF16提交后1次；第1段提交后1次|
|28|Y=R2+gate3*F2；本tile计算gate3|2048元素；336|后1次|
|退出|合法cache模式收尾；B14无候选写回|不纳入tile数|末尾1次|

普通路径合计G2=11912、G1=11984，比旧版均少2016。debug路径为外部核验保留MOD表，仍是13928/14000；计时使用普通路径。真实三KV原fixture分别为短G2、中G1、长G2；分组取决于有效历史尾块和实际CFG共享引用，不按“中KV”名称固定。KV长度改变attention内部K循环与耗时，多数其他tile数不变。不能把7候选说成一个Q112且可互相attention的序列。

主要来源：cross K RMS3584、cross K/V投影2688、GELU1568、三次残差1008，其余3068（G2）。总量大主要因为多种轻量算子也各自切成小tile，不等于有1.4万次大型GEMM。

普通路径全grid共38次（源码计数，不是profiler计数）：layout1+27数学阶段的27个出口+8个动态投影入口+FFN2额外中间同步1+退出1。debug额外物化MOD仍39次。各阶段内部另有CTA __syncthreads、128线程组barrier、cp.async等待等；这些用于行归约、shared装载和复用，并非全grid同步。

FFN2特殊：前96个CTA分别计算48输出tile的两个K段，后12个CTA在同步处等待；第0段先写BF16，第1段FP32暂存shared，grid同步后再按(FP32+bias)+第0段BF16合并写回。它不是把两段FP32直接相加后只舍入一次。

可以扩大调度任务而不扩大数学tile：例如CTA一次处理多行RMS或多个连续point tile，再统一领取/发布；当前阶段版静态分配已避免这些tile逐个原子领取。真正扩大GEMM tile则改变寄存器/shared、尾块和并行度，应分开讨论。合法候选融合如Norm+调制、Q/K RMS+RoPE、FFN1写回+GELU需要保留原数值边界；除本轮按需调制外，这些仍为规划候选，尚未实施。

源码入口见optimized/csrc/whole_layer.cu中的whole_layer、modulation4、point；FA2仍为fa2_stage.cuh。
