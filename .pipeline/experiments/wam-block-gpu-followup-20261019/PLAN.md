# M24 完整block GPU接续

- [x] 用户明确空闲即接续；登记新预算，原CPU/M22账本不改。
- [x] 本机无GPU进程/0显存，NVML仍24–25%异常；只允许精度使用，原≤5%计时门不降低。
- [ ] 六个真实action block0 × cached/uncached：R0自复核、R1、B0_TORCH/B0_ALIGNED及B14 guarded强基线严格输出/全部有效KV/元数据/输入/RNG与旧GPU gold核验。
- [ ] 仅资源门通过才做原A/A≤2.5%及ABBA/BAAB三轮计时；异常则留PENDING，不编造时间。
- [ ] 自有退出/累计预算/失败、两机晨间事实、HW及main镜像归档。

新范围：单A100/CPU≤2、累计生命周期≤7200秒、block≤5000次、新增≤1GiB，最多两次有据实现修复。所有尝试/准备/等待/清理计预算；绝对截止见AUTHORIZATION。仅一层真实block，不加载全模型，不开机器人/训练/生产patch/scheduler。原四false、旧FAILED/门、tile及保护源不变。NVML异常只撤销精度任务不必要的计时前置阻塞，不将未通过的计时资源门改PASS。
