# 三个小投影split-K：真实分派、数值顺序与调度CPU审计

本报告仅CPU只读审查与有限调度模型；未运行GPU、未连接SSH、未修改optimized源码。实际CUDA精度、发布可见性和性能由主作业验证。

## 真实SM120分派：不能依据旧函数名判断

冻结A中的 `whole_layer.cu:195–200` 在 `p.candidate && m==224 && n==3072 && k==3072` 时调用 `self_qkv_projection8`，后者调用 `self_qkv8_tile`。本轮三个单独阶段为：

| 阶段 | 实际输入/输出 | 权重索引 | 原活动入口 |
|---|---|---:|---|
| Self输出投影 | SA → SO | 9，bias10 | linear → self_qkv_projection8 → self_qkv8_tile |
| Cross Q | N2 → CQ0 | 13，bias14 | 同上 |
| Cross输出投影 | CA → CO | 19，bias20 | 同上 |

`action_projection`虽然存在，但上述SM120 B14分派没有调用它；它的4warp/两段实现不是本轮数学基线。`self_qkv_projection_group`处理Self Q/K/V共288个输出tile，同样调用原tile函数，但本轮不改变这组三投影。附近“两段K1536”“其他投影4warp”注释过时，不能用作SM120实际契约。

实际活动配置见冻结A `csrc/cutlass_gemm.cuh:70–88,202–284`：输出tile64×128、K64、三级cp.async、八个warp均计算MMA，warp tile32×32。真实形状 M224、N3072、K3072。

## 必须保留的数值次序

SM120十个K分段：p0…8分别K320，p9有效K192。迭代次数分别5次K64、末段3次K64。每段自身FP32 MMA归约顺序、mask和padding保持原函数。

设段内FP32部分和为Pp，BF16前段结果为Rp：

- R0 = BF16(P0)。
- p1…8：Rp = BF16(add_rn(Pp, FP32(Rp−1)))。
- p9：R9 = BF16(add_rn(add_rn(P9, FP32(R8)), FP32(bias)))。

最后一段是先加前段，再加bias，不能换序；每段都保留BF16舍入，不能将十段FP32部分和一起归约后只舍入一次。SM80另有两段和不同bias位置，本轮不能误改其契约。

## Torch 480 CTA 与自有960任务的区别

直接读取已有原始 `precision/results/partitions-data/{selfout,crossq,crossout}-trace.json`：三者kernel均为 `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8`，grid `[16,3,10]`，每block128线程、226寄存器/线程、65536B shared。这里480为总launch CTA数，不是同时驻留数，trace中的估算occupancy 0不能解释成实测occupancy为0，kernel名字也不能单独证明实际有额外ReLU。

| 路径 | 输出tile形状 | 输出tile个数 | K分段数 | launch CTA / 动态计算任务 |
|---|---|---:|---:|---|
| 已有原Torch trace | 128×128，K32四级 | 48 | 10 | 480个launch CTA |
| 冻结自有A | 64×128，K64三级 | 96 | 10 | 96个输出任务，每CTA内顺序10段 |
| 本轮自有候选 | 保持64×128，K64三级 | 96 | 10 | 960个K段任务，由170常驻CTA领取 |

因此，本轮只对齐K段可并行的组织，不声称已经复制Torch的输出tile、warp数、流水级数或CTA数。960不是960个同时驻留CTA，更不能写成“与Torch相同480任务”。三个阶段共2880个有效K段任务，各自重新初始化并执行。

## 领取顺序与无死锁条件

拟采用单队列 `task = partition * 96 + tile`，每次只能领取一个任务。任一p>0任务被领取时，其前驱任务号小96，因此已经获得owner或完成。等待仅指向相同tile的p−1，严格递减，不能构成等待环。最小未完成partition的owner无需等待更晚任务，可以前进；最终所有段完成。

证明条件：所有参与CTA驻留且获得执行进展、owner不能在未提交时退出、不重复领取、无硬件或算术异常。这里170个CTA领取960个任务。不能改成普通非cooperative的960block launch并沿用此自旋协议。

计数器初始化必须在本阶段起始grid同步前完成；三次阶段复用必须等待上一阶段grid同步完成。领取顺序用CAS或atomicAdd都可唯一化，若用atomicAdd退出探测，队列最终值允许超过960，不能把该原始计数直接当有效任务数；必须按实际实现确定终态口径。

结果发布要求：本CTA各写线程完成BF16输出写入 → 每写线程device fence → CTA barrier → leader device-scope release store(commit,p+1)。后继leader acquire读commit达到p → CTA barrier → 其他线程读取前段结果。仅leader fence不覆盖其余写线程的输出。当前stage内不同输出tile不共享输出地址。

## 有限CPU模型

`schedule_audit.py`对170个CTA、96输出tile、10个K段运行320个随机调度，全部完成，约1.714秒单进程。最长47轮，最多71个CTA同时等待；每次领取检查前驱已经拥有owner、任务无重复、96个commit最终全为10。反向partition顺序负控10轮后170个CTA全等待，正确暴露目标死锁类型。

模型没有执行CUDA，不模拟真实GPU弱内存、occupancy、warp调度、数学输出或速度，也没有声称模型通过证明全部实际CUDA输入。JSON结果为 `SCHEDULE-AUDIT.json`。

## 实现后复核

等待实现者完成源码后补充实际队列、计数初始化和发布路径行号；上述数学与调度条件已经核对原活动入口，不能在此之前把候选实现写成已审计通过。
