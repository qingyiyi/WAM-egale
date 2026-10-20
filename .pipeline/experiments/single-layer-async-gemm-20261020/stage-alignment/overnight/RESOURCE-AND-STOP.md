# 本轮资源与停止状态

更新2026-10-21T02:34:36.439687+08:00。本轮5970/6100完整block调用、13572/14000stage调用；
GPU worker生命周期619.984秒，CPU准备/编译进程3980.981秒。GPU生命周期含初始化与失败，
不是CUDA active；后者UNKNOWN。此前M39约665.866秒，合计约1285.849秒/
10小时；本夜GPU帽3小时未改变。远端本轮证据432.40MiB。
所有记录的owned子进程已退出；终态查询两张GPU均无计算进程。未终止他人任务。

停止/撤回：N256、point8192、强制内联、compact B14；grid_constant重复方案取消0GPU。
ready旧试验保留但已从活动实现撤回，按用户新要求不再新增调度。
当前活动源码恢复并重新构建，远端/本地九个关键源文件小文件摘要一致。

compact相关失败全部保留：CPU CUTLASS别名不匹配；资源metadata断言；
FA子空间大小误记；脚本tuple key；阶段DSO加载路径错误。
后者GPU1.566秒、0 block/0stage，错误build-lib覆盖了stage_final的可重建DSO，
其旧源码、构建日志、JSON原结果未改，不能声称旧二进制仍原样保留。
新stage_compact_r2复用这次实际构建的compact二进制，来源显式登记。

本轮得到有限精度通过和未速度对齐结论，不代表所有研究完成。
四质量/正式性能权限仍false。后续只针对阶段差距推进，尚未启动新作业。
详见[结果](RESULT.md)、[逐job终态](TERMINAL-REMOTE-RECEIPT.json)。
