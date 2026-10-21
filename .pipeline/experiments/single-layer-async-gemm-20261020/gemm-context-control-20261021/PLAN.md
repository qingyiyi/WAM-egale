# 同前序访存下的GEMM有限诊断

用户已同意执行Cross K/V→FFN→就绪tile的顺序。本作业先完成前两项归因，不改变数学或活动源码。

- 五个阶段：Self Q控制、Cross K/V、FFN1/2；三KV、两个独立进程、每格8轮。
- A为冻结native阶段；B为当前自有阶段。hot是固定输入独立热重放；prefix在A/B各自调用前执行相同native前序计算，真实传递全部前序中间量，prefix耗时不计入被测阶段。
- C为当前插桩完整层内阶段，固定60次完整预热。其前序实现与native-prefix不同，不把B/C差值直接写成纯同步。
- 记录全层/阶段原始globaltimer、cycle64、NVML时钟、编译资源、数值与有效KV。cycle/ns仅作为诊断量，不擅自等同某条硬件counter。
- 原接口、输入/有效KV/RNG有限精度保持；结果失败时保留并停止，未自动重跑。
- GPU生命周期≤600秒，CPU2，新增证据≤300MiB；绝对截止以JOB.json为准。旧M39账本累计，不重置旧截止。
