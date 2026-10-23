# 本轮资源与停止报告

终态：CLOSED，2026-10-23T16:26:34.464876+08:00。26个登记job；全部owned子进程及guard按PID/出生身份核验已退出，GPU0无compute进程、显存11MiB；GPU1原机器人队列不修改。

- 本轮保守占用窗口：2809.990秒 = 46.833分钟，低于120分钟。
- GPU worker生命周期：290.455秒 = 4.841分钟，包含失败/加载/退出；不是CUDA active。
- CPU编译job：758.377秒 = 12.640分钟，包含首失败。
- 其余窗口墙钟：1761.158秒，含准备/传输/用户交互/文献；没有分项active测量。CUDA active = UNKNOWN。不能把这项伪称GPU忙碌。
- 原累计18427.875秒；本轮保守闭合后21237.866/36000秒，剩余4.101小时。旧起点/截止不重置；本轮提前收口，不消费剩余额度。
- 远端新增研究目录209.307MiB，低于3GiB；无新增模型/episode大文件。
- 失败：T2-BC-build临时目录相对路径失败，1.704573秒CPU、0GPU；T2-diagnostic对未初始化allocated尾部误比较失败，11.643361秒GPU生命周期。两原记录保留，新attempt修runner/有效范围，不放宽数学门。
- 全部job无timeout；没有重复机器人任务、训练、NCU提权、驱动修改、FA2重写或新DAG。

原逐job状态及真实returncode见RESOURCE-LEDGER.json；清理见OWNED-CLEANUP.json。没有把退出脚本写成租卡关机，本轮没有管理租卡实例。
