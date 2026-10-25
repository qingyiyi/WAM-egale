# FFN源码与同步带读：只读核对，未修改算子

当前采用A，文件`.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`，与CLUSTER2作业的A字节快照相等。C2在`CLUSTER2-ATTRIBUTION-20261025/C2.cu`，为NO_GO研究候选。以下为源码依赖判断，不是新的精度/性能通过。

## 数据流与数学

B14×Q16的Linear有224行，D3072/F14336。M3是BF16归一化调制后输入。FFN1 224×3072乘W1转置得到224×14336；加bias后先BF16舍入，再tanh近似GELU并BF16写ACT。FFN2 ACT乘W2转置至224×3072，最后门控残差输出。W1实际Torch权重[14336,3072]，W2[3072,14336]。

`action_ffn_pipeline`878–882：每个CTA先处理自己FFN1任务、再领取ready FFN2任务；二者之间没有grid barrier。全层末尾g.sync保证全部输出就绪。

## FFN2本节

819起：4个M64行组×24个N128列块×7个K2048段=672任务，M末组仅32行有效。每任务四MMA warp3–6共同计算一个M64N128 tile的K2048部分和，32个K64主循环；不是一warp一段K。

579起`ffn2_partition64`：仅四warp进入WamMmaMultistage，acc为FP32，结束转存共享float result[64×136]。生成mainloop由`optimized/cutlass_gemm_codegen.py`从固定CUTLASS3.5.1生成，两个CTA barrier改为128线程named barrier。固定生成源码在`sm120-ffn-next-20261024/A-source/csrc/cutlass_gemm_mainloop.cuh`，665–666先cp_async_wait<0>再bar.sync2,128。

精确递推以611起commit64为准：令P_s为FP32 K2048部分和；r0=BF16(P0)，s1..5的r_s=BF16(P_s+float(r_{s-1}))；最后r6=BF16((P6+float(r5))+bias)。源码没有对每个P_s先单独BF16后再相加；不能把运行和的BF16提交口径误说成额外一轮独立partial舍入。最后从r6做gate乘法、再残差加法并BF16输出，debug=false不额外写最后F2。不能改成七段FP32连续相加只舍入一次。

## 屏障分类（仍未改代码）

|位置|依赖|判断|
|---|---|---|
|A608→A863|四MMA warp写shared，leader等待上一段commit，再全CTA消费|可优先研究只保留863，覆盖本地result ready与前序ready；须核全部调用点/128线程mainloop契约|
|A681→A866/867|每线程commit后fence、全CTA会合再leader release|681为可合并候选，保留867确保所有写线程发布完成|
|A848|leader写共享task，所有warp读取|保留；不能用仅warp内shuffle替代跨warp广播|
|A851|无ready任务时所有warp读完旧task后，leader才能改下一轮task|直接删有WAR风险，须改广播协议|
|A863|leader acquire确认上段提交后，所有线程读prev|合并后仍需一次会合或等价协议|
|A867|全部线程的F2写入/fence完成，leader才能release后继|当前发布协议需要|
|A870|leader release之后再全CTA会合|可研究与下一轮848/最后段publish695合并，不能先报已安全删除|
|A882|cooperative全部CTA输出完成|只领取完任务不等于计算完，当前层末尾需要|
|A775|FFN1主GEMM返回后、union变成result前|固定CUTLASS原始mainloop在665–666已wait0+CTA同步，外层为优先冗余候选|
|A782|四warp写一半结果，八warp读取后处理|当前shared消费需要|
|A808|一半result读完、下一半才能覆盖|第一半需要；最后半可能与publish695合并|
|A695|各写线程fence完成再leader增加ACT ready|当前发布协议需要|
|A703|leader发布完成后会合|可研究与后续领取task会合合并，需全部调用点核对|

C2区别：每段有8轮K256 slab，每轮8个K32子步。964的cluster.sync发布双CTA各自的64×128 ACT；974确保8warp写swizzled A完成再四MMA读取；983确保MMA不再读A/B槽才能复用；985确保所有peer读完ACT才能覆盖。每段内循环64×2=128 CTA屏障，另16 slab cluster屏障（不含job尾）。单A缓冲不改时974/983不能直接删。W2有双buffer，ACT的swizzled A仍单buffer，不能据W2双buffer就判所有交接可省。

C2当前不是专职TMA warp：thread0发W2 TMA，全部8warp参与ACT复制与commit，warp3–6做MMA。全部线程目前都调用projection_wait；减少等待参与者/提前下次提交需要保持slot生命周期，不是删__syncthreads一行。

## 概念与验证边界

__syncthreads是CTA内到齐与内存顺序，不是全GPU屏障；g.sync是cooperative grid，cluster.sync仅当前cluster；threadfence是调用线程的内存顺序，不等待别人；mbarrier可拆arrive/wait并跟踪异步事务，但不是自动等价替换。官方参考：https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

源码冗余候选不代表SASS仍保留两条屏障或必有速度收益。后续若执行，保留独立A，先有限数值/七段/有效KV/RNG，再三KV无插桩完整层；不一次删除所有屏障。本次只读带读没有新增GPU算子实验或改变采用版本。
