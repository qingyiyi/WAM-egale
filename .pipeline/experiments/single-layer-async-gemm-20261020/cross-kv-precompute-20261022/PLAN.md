# Cross K/V预计算与逐阶段计时

用户已批准本轮执行。沿M39与原累计十GPU小时，当前已用7699.251秒；准备工作CPU先行。新GPU生命周期最多2700秒，实际开始前登记绝对截止，预留300秒清理。GPU0独占、CPU2线程，未使用本机GPU；空间本地768MiB、远端256MiB。旧截止与原记录保留。

- [x] 明确公平缓存：自有/guarded均读取最终K（含原RMS）和V；原始K额外保留作审计。
- [x] 实现独立context缓存参数，跳过两投影和K RMS；普通与缓存路径共用其余阶段。
- [x] CPU编译正常版与独立诊断版；所有失败如实保留。
- [x] 9签名、27结构、三KV原Torch缓存对照，输出/中间值/输入/缓存/RNG；有限门后计时。
- [x] 两独立进程、七臂、每臂600预热/24交错轮，短/中/长完整层。
- [x] 独立插桩三KV×两缓存模式，25阶段及layout/exit、每CTA grid等待；报告插桩扰动。
- [x] 采用或退回、owned清理、完整阶段表、HW与main镜像。

七臂：原Torch静态warm final cache；guarded同分组uncached；guarded原形状projection-only缓存；guarded final缓存；上一版自有分组uncached；本版自有分组uncached；本版自有final缓存。冷prepare host时间、baseline额外扩展时间和缓存容量单列。

精度门保持逐位；缓存未改变字段也纳入输入不变审计。计时仅完整单层Graph；不把profile数值相加冒充正常层时间，不把CTA平均等待当作墙钟占比。全部大模型/机器人/四科学权限不变。

终态：两机研究入口采用，完整结果与阶段诊断已收口；两CPU准备失败及首profile失败保留，GPU已清理。
