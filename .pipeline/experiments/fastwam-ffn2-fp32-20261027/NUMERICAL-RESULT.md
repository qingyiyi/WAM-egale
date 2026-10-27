# 数值结果
官方eager为差异参考，compile单独列。relative L2=||candidate-reference||2/||reference||2；maxabs为对应张量坐标单位，不是统一物理误差。
|代表层|B7 FFN2 relL2|FFN2 maxabs|整层 relL2|整层 maxabs|
|---|---:|---:|---:|---:|
|0|0.435998%|0.25|0.342193%|0.0625|
|15|0.459892%|0.0625|0.074005%|0.03125|
|29|0.455003%|0.015625|0.050421%|0.125|

B1三层全部字节一致。B7非FFN2/末残差范围全部字节一致，输入/RNG不变。layer15 compile/eager整层relL2=.305806%、maxabs=.125；本候选=.074005%、maxabs=.03125。不能将这个单层对比推广为所有层更安全。
同一真实layer15 FFN2输入，以FP64矩阵乘加为数学参照，严格/eager relL2=.426638%，候选=.166812%。候选较接近数学乘加，并不意味着较接近原机器人策略。
原严格递推：每段j累计256个K元素，z_j为FP32 MMA；r_j=BF16(z_j+FP32(r_(j-1)))，末段加bias后BF16。候选完整K4096维持一个FP32累加器，末尾加bias、一次BF16。不是乱序atomic，也没有修改RMS/LN规约。
原始：remote/compare-attempt02/results/PRECISION.json、WHOLE.json。首轮借用了可写reference导致污染，原始保留但排除；修订使用独立override副本并增加reference未变检查。
