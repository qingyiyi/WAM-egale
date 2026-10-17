# 完成当前case的资源交接控制

首轮v2仅等待同时最后臂的窗口，在480秒后NO_CHANGE，原2worker继续，旧终态不改。此独立控制器仅暂停已birth-pinned的原CPU runner；client/server当前整局继续，再用原v1 Runner.spawn/environment与原client_entry执行同一已登记case缺少的后续臂。没有新case/新seed/重试episode，没有模型数学/输入协议修改。原FINISH全部drain、manual owned client reaped后退出空闲server，恢复原runner写自己的真实PARTIAL/清理。v2只读复用所有完整原三臂组。

CPU控制上限90min/父协调2h，均计入原48h不重置。M16 forward/存储系列cap不变（计划实为6001000forward/10GiB）；原两GPU继续完成已有任务，直到清理才启动新4worker。控制source与eval source分别登记；无smoke、测试、模型探针或SHA。
