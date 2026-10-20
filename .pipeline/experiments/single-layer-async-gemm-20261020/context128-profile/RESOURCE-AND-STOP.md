# 当前profile资源终态

新请求下独立登记，原20:20:28、旧1497/1500与372/420账本保留。

- 插桩profile：12完整block、150独立Torch Linear；GPU worker生命周期11.383秒；CPU编译119.095秒。
- 冻结native完整层profile：12完整block，GPU进程及guard约11.810秒。
- 合计24/24完整block、150/150原语，GPU生命周期约23.2秒/900秒，均正常终态；没有NCU kernel replay。
- NCU查询缺`--metrics`的语法提示后修正查询，得到ERR_NVGPUCTRPERM；GPU硬件counter未知，不重试绕过权限，不改驱动/时钟。
- 两个LAUNCH辅助JSON最初多出字面量换行转义，原文本保存为.raw后仅规范化JSON尾部空白；没有重启作业、改PID或补造旧清理收据。
- 原源码/当前production DSO未改，插桩与trace在本profile目录；末态进程身份和空间见FINAL-AUDIT。
- 无算子后继GPU作业。GPU1的RoboTwin剩余42条是独立获准队列，当前运行不受本profile清理影响。
