# 既定Object评估的双卡并行派发

用户最新明确要求两卡各并行eval，核实自动接续后主要转向本机算子。只提高既定1500局吞吐，不新增科学臂/重复局，不改投机模型、训练、接受器或现有评估源码。

- [x] 确认原队列已有真实自动接续；每卡约14GiB，空闲约26GiB，CPU quota45核/内存180GiB。
- [x] 阅读原dispatcher/launcher/guard及端口、CPU与数据隔离规则。
- [x] 以实际READY GPU UUID、PID出生身份、存活CUDA设备FD、持续teacher计数及NVML上下文集合建立唯一映射。不能直接把宿主PID猜成容器PID。
- [x] 在旧CPU派发器的sleep检查点交接；现有model/client及独立guard不暂停。
- [x] 最多每卡两个独立model worker，原端口/CPU/输出/完整episode隔离；保留原M17起点/截止、remaining_forward_budget和15GiB全局cap。
- [x] 验证四worker有实际完整episode终态、后继分片自动派发，汇总吞吐与异常；若无法唯一归属则保持原两worker，不放松准入。

运维代码/记录在独立目录，远端只写~/cxz/ops/object-parallel-20261019及原已授权运行目录的动态状态/尚未开始分片。冻结投机源码不改，source_code_commit与selected、tau、reset bank、case身份、评估种子与同case同GPU约束保持。

新运维设置CPU≤2，代码/审查≤90分钟、持久化≤16MiB；不新增GPU实验预算。所有原phase继续使用原独立guard，原绝对stop/cleanup不变。旧CPU controller交接记录保留真实SIGSTOP/SIGKILL及PID出生身份，不伪造自然exit0。交接前失败则SIGCONT恢复旧CPU controller，期间不启动新episode；交接后出错只请求当前完整episode收尾，保留PARTIAL，不自动重试。

本轮并发是质量/探索性成本运行，不能作为独占正式性能结果。四项false保持。上下文归属映射是资源准入证明，不是算法初态/首两段配对证书。
