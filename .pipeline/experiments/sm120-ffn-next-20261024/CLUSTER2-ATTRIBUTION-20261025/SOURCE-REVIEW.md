# CPU源与进度核验

B2设备体继承A，只改cluster2 cooperative启动与容量查询。C2由冻结C4保持M64×N128×K2048、四MMA、64个K32 helper和七BF16提交；job336×rank2覆盖原672任务，part=job/48，row=(job%48)/12，ng=job%12，tile=row*24+ng*2+rank。每job八轮K256，每CTA全局读自身M64×K128，组内共享；G2只将peer读取改global ACT，原准备/屏障保留。

所有cooperative CTA均resident，按partition-major有限递增job领取；等待的前序job编号严格更小，任一owner先完成自身FFN1才进入FFN2，前序F2 release经所有写线程fence后发布。cluster在覆盖DSM slot前完成reader屏障，退出job前两rank同步。未建立等待未领取后继的环。

数值：每个K2048仍64个K32、每个K32两次原K16 MMA，原独立FP32 partial按七段BF16舍入/有序提交；padding行显式零，与有效行分开。旧E-CLUSTER-PAIR为M16整ACT驻留，不能作为本C2单变量对照。

静态差异：C2/G2均以单个swizzled A片串行交接，每个K32有两次CTA同步；它们的W2双缓冲与A的CUTLASS三级A/B流水不同。是否构成主要耗时由本轮对照/插桩决定，不先写结论。硬件计数器不采，动态流量未知。
