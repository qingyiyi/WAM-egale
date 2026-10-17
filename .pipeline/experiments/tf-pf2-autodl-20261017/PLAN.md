# TF_ON连续/PF2双卡完整DEV执行

> 终态：六对十二局已完整、CPU审计与自有进程清理完成；投入NO_GO，研究仍active，四false不变。当前结果以[RESULT](RESULT.md)为准，下方时间条目保留执行历史。

本轮用户已批准源码实施、迁移及六pair双卡执行，并将AutoDL资源许可改为12h窗口；总5GPUh被替代，原12局/49000forward/3GiB与失败止损保留。T0为2026-10-17 11:11:41北京时间，23:11:41清理硬截止，22:56:41停止新工作。准备≤2h计入窗口；不得重置旧预算/结果。

- [x] 独立源码实现（本地静态审查，不冒称运行时通过）：同wrapper C/P、无SHA loader、真实S/T计数与审计、无smoke/独立测试。
- [x] 必要环境/单份资产迁移，记录容量与生命周期；不改系统服务/他人进程。
- [x] 三波固定六pair自然完整任务，严格初态与前两S隔离、全局止损。
- [x] 联合成功/长度/800罚分/实际成本审计，不以调用数替代速度；成本可比性UNKNOWN，投入NO_GO。
- [x] 自有进程清理、1267原件回传与最终HW/报告；四false与旧NO-GO保留，实例未关机。

登记与精确输入：`投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r1/PREREGISTRATION.json`。无需重复询问已批准的执行范围，不训练/扩大case/独立TEST/正式性能。

2026-10-17 12:03：独立源码已冻结并上传，执行commit `c91c649f15ad4037690e88395b8fa0cb9ff680ea`，父版本仍4f04。七权重固定发布revision直下完成，456.15s；原环境3/4压缩流完成。新完整episode0，GPU worker尚未启动。来源见job的SOURCE-FREEZE.json、SOURCE-UPLOAD-RECEIPT.json及MODEL-DOWNLOAD-RESULT.json。

2026-10-17 12:41：部署实际完成，GPU执行器PID12445已启动并在等两服务真实ready。最终执行源码f4c0031429e5c5fc468770cfeaf3b5510b0b92d2，替代未运行c91：独立shared-prefix准入只将隐藏SHA guard换原始字节比较，其余旧check与数学不动；kernel/flag命名显式锁定prefix目录。环境源流DNS失败保留，余下249MiB通过已有SFTP连接补齐；原准备时钟未重置。新完整episode仍0。

2026-10-17 12:51：r1模型导入前失败并清理；r2两服务READY，但两client在环境初始化前包路径错误、0环境步/0targetforward，FAILED/STOP/收据保留。修复导入路径另建r3，源码9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a；原T0、六pair/12完整/49k/3GiB总边界不重置。r3两卡真实完整任务已开始（1/29 C、1/36 P），优化reset真实请求/RNG检查通过，尚无完整自然终态。
