# 资源、预算与停止

本轮授权为用户明确要求实现Cross K/V复用、先精度后计时。新窗口2026-10-22T16:05:43.031227+08:00至2026-10-22T16:35:43.031227+08:00；旧预算起点/截止均保留。实际2026-10-22T16:09:21.843008+08:00已核对全部owned PID及出生身份退出，无需发送kill信号，两卡compute-app列表为空。

CPU编译298.519秒，GPU隐藏，不计GPU活跃时间；五个GPU worker总生命周期111.688秒。含上传、等待、清理的整窗保守计费218.812秒，累计7699.251/36000秒，剩余28300.749秒。worker包含在整窗内，不重复相加。CUDA active时间未独立采集，UNKNOWN。

五步全部exit0、无timeout或失败。9签名81次、27结构108次、grouped baseline6次，两计时进程各13170次完整层调用；是算子研究重放，不是机器人episode。本地证据及构建33.40MiB/512MiB，远端11.35MiB/128MiB。

[JOB](JOB.json)、[CPU构建](BUILD-STATUS.json)、[作业终态](STATUS.json)、[清理](FINAL-CLEANUP.json)、[本机采用](DEPLOYMENT-LOCAL.json)、[远端采用](DEPLOYMENT-BAAI.json)。本机未做新GPU计时；本轮仅在远端GPU0测量。不自动接续过期RoboTwin或新的机器人任务。
