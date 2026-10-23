**T1 完成｜2026-10-23T15:46:17.476063+08:00**
本次评估：FFN2同tile准备方案与存储核查；尚未采用。当前实际保留：**A**。
源码 `fb650be4d78b876b980e0866117eae21db894139aa8b7cfb5c351ba2121a1717`；独立二进制 `a41d778d31f8a72fb9255c2152db584f2b0c3514d3f641c6bddba826d4f06473`。
A100-SXM4 GPU0；B14/Q16/D3072/H24/FFN14336；KV288/1152/2160；双方最终Cross K/RMS/V预计算。108CTA×8warp；shared112384B；255寄存器/线程；静态local1016B；1CTA/SM。结构来自源码核对；阶段耗时本轮未插桩。

|ID／阶段|数学工作与形状|tile与任务数|warp分工|同步与数据交接|
|---|---|---|---|---|
|01 layout|有效历史slot压紧|按pool/256轮；CTA0|8W|CTA0搬运后grid1；slots就绪|
|02 调制参数|table+temb，FP32|消费处生成；无独立tile|消费warp|无独立屏障|
|03 LN1|一行D3072，Welford LN|224行tile|W3–6归约，8W搬运|行内barrier；grid1，N1就绪|
|04 调制1|BF16[N1×(1+scale)+shift]|2048元素×336tile|8W vec4|grid1；M1就绪|
|05 Self QKV|3×[224,3072]×W；各K1536×2|64×128，合计288tile|8W cp.async/MMA/写回|队列广播/两段提交；grid2|
|06 Q RMS+RoPE|两行D3072 RMS后复数旋转|112tile|两组4W|各组归约＋CTA；grid1|
|07 K RMS+RoPE|两行D3072 RMS后复数旋转|112tile|两组4W|各组归约＋CTA；grid1|
|08 private KV|候选suffix私有，历史只读|无插入tile|无|无；由attention分别寻址|
|09 Self attention|FA2 Q64/K64/D128；有效Q16/32|96/168/96tile；KV循环5/18/34|两组4W；QK/PV MMA|FA2组内barrier；grid1|
|10 Self输出|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|11 残差1|BF16[x+gate×SO]|2048元素×336tile|8W vec4|grid1；R1就绪|
|12 LN2|D3072 LN＋affine|224行tile|W3–6归约，8W搬运|行内barrier；grid1|
|13 Cross Q|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|14 Cross K|context×Wk，warm预计算|warm不执行；cold192tile|warm无|warm无|
|15 Cross V|context×Wv，warm预计算|warm不执行；cold192tile|warm无|warm无|
|16 Cross Q RMS|两行D3072 RMS|112tile|两组4W|组归约＋CTA；grid1|
|17 Cross K RMS|最终K已预计算|warm不执行；cold512tile|warm无|warm无|
|18 Cross attention|Q16×K512×D128；KV64×8轮|168tile|两组4W FA2|FA2组内barrier；grid1|
|19 Cross输出|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|20 残差2|BF16[R1+CO]|2048元素×336tile|8W vec4|grid1；R2就绪|
|21 LN3|D3072 LN，FP32输出|224行tile|W3–6归约，8W搬运|行内barrier；grid1|
|22 调制3|BF16[N3×(1+scale)+shift]|2048元素×336tile|8W vec4|grid1；M3就绪|
|23 FFN1|D3072→14336；原BF16边界|108×128×256/K32；12×32×256/K64|主8W MMA，尾4W；8W写回|组生产28/28/28/36 release；无grid|
|24 GELU|BF16[tanh-GELU(BF16 FFN1)]|并入23，无额外tile|8W CUDA core|同23；ACT发布后可读|
|25 FFN2|14336→3072；K7168×2|128×128；48输出/96分段tile|W3–6 MMA；8W提交|mainloop bar2/128；CTA；原partial acquire|
|26 残差3/FFN出口|BF16[R2+gate×BF16 F2]|并入第二段FFN2提交|8W CUDA core|partial/输出release；最终grid1|
|27 最终退出|B14无cache回写|无计算tile|108CTA全体|grid1，返回输出|

循环内同步沿原实现：LN保留Welford树及行内屏障；GEMM保持原K循环与cp.async等待；FA2保留组内barrier。没有把动态barrier次数猜成常数。

|阶段／可比区间|自有短/中/长|Torch短/中/长|Torch/自有|测量状态|
|---|---|---|---|---|
|01 layout|—|—|—|NOT_MEASURED|
|02 按需调制|—|—|—|NOT_MEASURED|
|03–04 LN1/调制|—|—|—|NOT_MEASURED|
|05 QKV|—|—|—|NOT_MEASURED|
|06 Q RMS/RoPE|—|—|—|NOT_MEASURED|
|07 K RMS/RoPE|—|—|—|NOT_MEASURED|
|08 private KV|—|—|—|NOT_MEASURED|
|09 Self attention|—|—|—|NOT_MEASURED|
|10 Self输出|—|—|—|NOT_MEASURED|
|11 残差1|—|—|—|NOT_MEASURED|
|12 LN2|—|—|—|NOT_MEASURED|
|13 Cross Q|—|—|—|NOT_MEASURED|
|14–15、17 Cross K/V/RMS|—|—|—|NOT_EXECUTED（warm预计算）|
|16 Cross Q RMS|—|—|—|NOT_MEASURED|
|18 Cross attention|—|—|—|NOT_MEASURED|
|19 Cross输出|—|—|—|NOT_MEASURED|
|20 残差2|—|—|—|NOT_MEASURED|
|21–22 LN3/调制|—|—|—|NOT_MEASURED|
|23–26 FFN共同区间|—|—|—|NOT_MEASURED|
|27 退出|—|—|—|NOT_MEASURED|

阶段值未测，不借用历史数字。Torch独立热Graph与kernel内区间的缓存/边界不同，阶段值不可相加替代完整层。

|KV|A ms|本次变体 ms|A/变体|D/变体|G/变体|波动与结论|
|---|---:|---:|---:|---:|---:|---|
|288|0.871910|未运行B/C|—|—|—|A为T0已测；T1无GPU测量|
|1152|1.043379|未运行B/C|—|—|—|A为T0已测；T1无GPU测量|
|2160|1.056000|未运行B/C|—|—|—|A为T0已测；T1无GPU测量|

唯一方案：7KiB独立prep准备首8行及bias/table，原8warp提交；B/C同二进制只改顺序。准备量有限，不预设正收益。当前仍A；进入T2实现。
