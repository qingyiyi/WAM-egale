# 资源与停止报告

当前状态：COMPLETE_BOUNDED_SCOPE_ALIGNMENT_NOT_ACHIEVED；本机/BAAI本段全部owned已退出，真实PID出生时间收据见两个FINAL-CLEANUP文件。

本段保守计费531.767秒；原M39累计6004.497/36000秒，剩余29995.503秒。原09:00截止不重置。
远端GPU worker生命周期323.265秒，本机41.044秒；本机等待/失败保守计费208.502秒。CPU编译/准备952.151秒单列，CUDA active时间未测。

|namespace|终态|计费秒|GPU worker秒|CPU准备秒|
|---|---|---:|---:|---:|
|01-forceinline-attention|COMPLETE_ATTEMPT|81.945|81.945|147.7709755897522|
|02-specialize-attention-route|COMPLETE_ATTEMPT|81.847|81.847|144.87197732925415|
|03-complete-device-lto|FAILED_OR_PARTIAL|0.000|0.000|1.6657531261444092|
|03b-lto-target-fix|COMPLETE_ATTEMPT|83.347|83.347|316.1005835533142|
|local-ncu-current|STOPPED_BEFORE_GPU_ADMISSION_POLICY|90.601|0.000|含于本机保守计费|
|local-ncu-reference-fixed|COMPLETE|31.587|19.489|含于本机保守计费|
|local-ncu-stages|STOPPED_NO_EXCLUSIVE_WINDOW|52.768|0.000|含于本机保守计费|
|local-ncu-visible-idle|FAILED_OR_PARTIAL|33.545|21.555|含于本机保守计费|
|lto-stage-diagnostic|COMPLETE_ATTEMPT|76.126|76.126|341.74134063720703|

停止方向：FA2强制内联无稳定收益；路由模板专化无收益；全局LTO稳定退化，不保留三项。原上限2→3的前瞻修订和CPU构建兼容失败保留，不重写旧预算/FAILED。不是因预算耗尽停止，也不把余额自动挪给RoboTwin。

新增本地证据153.7MiB、远端93.9MiB；非新增模型权重。没有租卡/关闭租卡的动作。

RoboTwin原10/22 01:26截止已过，当前没有本机或远端本项目机器人worker；未启动1800局。
