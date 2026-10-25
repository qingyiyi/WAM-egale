# FFN1 双CTA共享权重：有界实验

当前采用 S；冻结 W 只作实验起点，旧结果不改。使用 HW experiment 记录本轮。

- [x] 授权、GPU空闲、S/W源码与独立二进制、三真实fixture核对。
- [x] 实现 L（仅cluster+行配对）和 D（额外共享B）；CPU审查槽生命周期。
- [x] 独立编译；完整真实block精度、七段、输入/KV/RNG、计数。
- [x] 两进程×短中长KV，S/W/L/D/原Torch同轮无Graph；插桩分列。
- [x] 完整18阶段表、采用决定、回传、清理、HW/main。

预算/绝对截止见JOB.json。100分钟含准备等待失败清理、9000完整block调用、2GiB、4CPU；不继承或重置旧W作业。

224个cluster任务，每个两CTA，各计算M64N128输出；不同M、同N。
85cluster/170CTA，双槽仍K64。D中rank0 TMA A+B、rank1 TMA A；
rank0将16KiB swizzled B异步复制到rank1的同布局local槽。原ldmatrix仍读local。
rank1先发布目的槽credit，收到A+B后ACK；rank0 EMPTY同时等四MMA读完与ACK。
只在初始化/排空做cluster汇合；K热循环用mbarrier，无每K全cluster屏障。
每对每K逻辑global输入48KiB→32KiB，另加16KiB DSM；这不是实测HBM下降。
MMA工作量和输出均不翻倍；不改FFN2，不做TMA输出store或二维输出扩展。

L控制cluster启动/任务映射；D-L定位共享路径净效果，D-W与D-S检验整层收益。
通过有限fixture不等于全输入/机器人质量；四科学权限仍false。

结果：D未采用，保留S。完整层、等待定位、原始数据、清理与发布收据分别见本目录；GitHub以PUBLICATION.json为准。
