# FFN1共享A对照：本轮结果

## OBSERVED

判定：COMPLETE_NEGATIVE_RETAIN_S；保留S。共享A/共享B/S/W/L_A均在两个独立进程、同三fixture中交错正常计时，阶段观察器另列。

共享A让两个CTA负责相同M64行、相邻两个N128列，各保留独立FP32输出累加器。rank0读A8KiB+B16KiB；rank1读自己的B16KiB，A由DSM复制进入其local槽；原ldmatrix仍读本地shared。双槽、ready/empty/目的credit/完成ACK不变，保留完整48×K64、bias→BF16→GELU→BF16与FFN2七段数值顺序，不删除FFN阶段grid同步。

三KV/两进程的27tensor、七段partial/running、有效KV/输入/RNG/非默认stream、实际角色/传输/提交计数、前后输出均有限逐位通过。控制L_A仅同cluster/同映射、每CTA各自加载A+B。

实际计数对应共享A的FFN1：A10752×8KiB=84MiB、B21504×16KiB=336MiB、DSM10752×8KiB=84MiB。软件global tile请求504→420MiB；共享B为504→336MiB加168MiB DSM。这里包含重复与OOB填零的逻辑tile范围，不能当实际HBM字节。

## HYPOTHESIS / UNKNOWN

共享A把每次复制缩小一半，但仍每K64需目的credit、复制完成、源ACK和MMA槽回收；不会自动减少交接次数。两版本的M/N配对不同，因此共享A与共享B的同场净差不完全等于复制大小的单因素作用。各自应与其独立加载对照比较。

A原始矩阵224×3072 BF16仅1.3125MiB，重复global访问可能由cache吸收；本轮无L2/DRAM计数器，缓存命中/物理流量UNKNOWN。不能声称省84MiB HBM。等待观察器若追加，见WAIT-ANALYSIS，不以等待总和代替整个kernel时间。

## 同预算等待诊断

[等待表](WAIT-ANALYSIS.md)：接收READY约1.03μs/K64、源EMPTY约1.76μs、MMA输入等待约1.35–1.37μs；bulk提交约0.023μs。观察器输出/七段/状态通过，整层扰动0.81%–1.00%。等待不相加，不等于纯DSM传输延迟。

同轮正常完整层：共享A比共享B延迟减少21.24%/20.15%/18.76%，但比S增加26.48%/24.20%/21.70%。共享A FFN1约323μs，同场W约152μs；先前B阶段约560μs，后者明确是上一job的阶段测量。L_A同轮基本与W接近，不能解释共享A的净退化。所有正常版本资源仍170CTA/100352B shared/255regs/784B local。

## RECOMMENDATION

完整层收益决定采用；共享A即使比共享B快，也必须赢过S才能替换。负结果时保留S，不自动扩大cluster、改tile或追加第三种设计。原B结果、失败和四项科学false不改；算子重放不等于机器人闭环/E2E。

## 固定入口

本目录`whole_layer.cu`：FFN1_SHARE_A=1为共享A，0为L_A。`S.so`、`W.so`、`B.so`均独立。原始两轮在`measure-1/RESULT.json`和`measure-2/RESULT.json`，阶段/速度汇总在STAGE-TABLE.md与SUMMARY.json，源码覆盖审计见SOURCE-AND-SCHEDULE-AUDIT.json。
当前保留S源码：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`。

## 收尾

5400次完整block调用，登记生命周期10.96分钟，remote证据172.14MiB，10个phase正常退出，无失败/timeout。S公开入口再次通过，15个关键源码/DSO/原tensor回传逐字节核验，无SHA；owned进程按PID出生身份确认退出，5090无残留，不关闭实例。共享B原job独立保留4836调用/17.09分钟，不重置其预算。
