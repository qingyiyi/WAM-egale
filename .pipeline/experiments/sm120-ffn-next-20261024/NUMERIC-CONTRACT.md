# 当前SM120 A：FFN精确数值递推

依据冻结A-source/csrc/whole_layer.cu的ffn2_partition64、ffn2_commit64和action_ffn2_splitk；generated mainloop另行固定。标量符号表示每个有效输出元素，不能将其解释成可任意重排的普通实数求和。

定义round_B为__float2bfloat16_rn（实际四元素路径为__floats2bfloat162_rn），add32为__fadd_rn，mul32为__fmul_rn，widen为BF16到FP32的精确转换。

每段独立acc.clear()，执行原CUTLASS WamMmaMultistage：32次K64迭代，各迭代保留原ldmatrix、warp mma.sync m16n8k16顺序。P_s为该段独立FP32 accumulator输出；不是以上一段BF16/FP32状态作为MMA初值。内部MMA是融合乘加；epilogue中的显式add32/mul32不合成FMA。

|s|hidden范围|MMA初值/段内顺序|精确BF16递推|bias/发布|
|---|---|---|---|---|
|0|[0,2048)|零；32×K64原顺序|R0=round_B(P0)|无bias；F2写R0，release提交计数1|
|1|[2048,4096)|零；32×K64原顺序|R1=round_B(add32(P1,widen(R0)))|F2写R1，release计数2|
|2|[4096,6144)|零；32×K64原顺序|R2=round_B(add32(P2,widen(R1)))|F2写R2，release计数3|
|3|[6144,8192)|零；32×K64原顺序|R3=round_B(add32(P3,widen(R2)))|F2写R3，release计数4|
|4|[8192,10240)|零；32×K64原顺序|R4=round_B(add32(P4,widen(R3)))|F2写R4，release计数5|
|5|[10240,12288)|零；32×K64原顺序|R5=round_B(add32(P5,widen(R4)))|F2写R5，release计数6|
|6|[12288,14336)|零；32×K64原顺序|R6=round_B(add32(add32(P6,widen(R5)),widen(b2)))|最后才加bias；最终OUT，release计数7|

末段gate=add32(table5,temb5)，OUT=round_B(add32(widen(residual2),mul32(widen(R6),gate)))。数学残差residual2与上表的running R_s是不同对象。正常A最后一段不写F2：F2仍是R5；debug才额外写R6。不能把正常F2读成最终FFN输出。

每段P_s先在四MMA warp寄存器，之后转存CTA shared result，全部八warp消费。段计算可早于前段提交，但commit必须acquire读到该tile的精确s，CTA同步后才能读取前序全局F2；各写线程threadfence、CTA同步、leader release s+1。下段消费前不能覆盖共享中间状态。FFN出口grid.sync等待所有输出完成。

FFN1主tile128×256/K32、八MMA warp，尾tile32×256/K64、四MMA warp。每输出完成全部K3072后：U=round_B(add32(acc,widen(b1)))，然后用FP32计算GELU(tanh近似)，A=round_B(0.5*U*(1+tanh(beta*(U+0.044715*U^3))))；beta等于源码的sqrt(2)*2/sqrt(pi)*0.5。GELU表达式的具体编译与FMA行为沿用原编译参数和源码，不能据公式改写。ACT为BF16 global，先发布所有有效写线程的写入，再由leader释放ready计数。FFN1无重算。

FFN1 hidden输出tile N256、ACT ready分组、FFN2数值K2048、FFN2内部K64是四种不同粒度。现A等待对应行的两组FFN1生产完成，才领取该行FFN2任务；每CTA先完成自己的FFN1任务，cooperative grid全部驻留，没有依赖尚未调度的FFN1生产者。

本轮若同owner承接相邻段，仍需独立零初始化下一段P、执行每一次round_B；不能把R直接作为下一段MMA初值，也不能仅在组边界舍入。已省略global状态的组内段只能由同owner消费，组外状态仅在组边界发布。
