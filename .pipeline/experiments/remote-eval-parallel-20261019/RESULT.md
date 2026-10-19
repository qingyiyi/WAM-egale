# 远端并行评估与自动队列核验

核验时间：2026-10-19T12:05:24.465035+08:00。两卡各两个独立model/client，均READY、PID出生身份匹配、有实际teacher返回和完整episode；新两worker各至少1局完整终态，非仅启动目录。已建立上下文集合唯一映射，未把NVML宿主PID直接当容器PID。

|GPU|分片|服务端口|完整局（此时）|
|---|---|---:|---:|
|0|FINAL_200_274|19900|221|
|1|FINAL_350_424|19910|12|
|0|FINAL_275_349|19901|1|
|1|FINAL_425_499|19911|1|

总去重完成435/1500局：T278、连续TF78、PF279；其中77组三臂齐全。匹配Target200已完整，剩余FINAL_0_99/FINAL_100_199在自动队列。原控制器完成T200后实际自动提交350_424，新运维派发器也已实际提交275_349和425_499，自动接续不是只凭计划推断。

每卡显存约28.5GiB，当前GPU92%/100%；CPU quota45核、内存180GiB，4组独立2+2线程及控制CPU保持原绑定。四路并发不等于耗时减半；尚需更长并发吞吐序列来重估ETA，暂不把启动阶段速率外推。队列保留同case同GPU分配，因此尾部可因分片分布变为少于四路，不为占满GPU重复已完成任务。

运维交接仅暂停并清理旧CPU dispatcher（真实SIGSTOP/SIGKILL，出生身份匹配）；现有model/client/guard无信号、完整episode未截断。旧controller原收据在远端ops下保存，当前动态CONTROLLER-LAUNCH指针更新为新PID566820，原旧退出不伪造exit0。新源码仅位于~/cxz/ops，投机算法、训练、接受、评估源码未改。每局仍用既定优化reset；所有原完整失败和PARTIAL保留。

原M17起点、forward总预算与15GiB cap不重置；原停新2026-10-21 16:56:31、清理17:11:31，北京时间。新dispatcher继续逐阶段用原独立guard，失败仅收当前完整episode，不自动重试。GPU资源并行墙钟属于探索性吞吐，不是正式E2E；四项false不变。

已完成历史套件：Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，均各500。Object当前面板未全，不外推成功率。严格首两S配对/数据曝光/独立性缺口保持原记录。

证据：[CHECK-120524.json](CHECK-120524.json)、[四worker归属](FOUR-WORKER-OWNERSHIP-RESULT.json)、[交接核查](HANDOVER-CHECK.json)、[运维源码](dispatch.py)。远端实际动态证据：~/cxz/ops/object-parallel-20261019及原Object v3运行目录。每5分钟既有CPU只读watch继续；不是Codex自动驻留。
