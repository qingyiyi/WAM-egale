# 资源与收口
关闭 2026-10-27T09:20:13.336457+08:00。实际runner生命周期合计 173.362s；保守按完整登记窗口 1304.832s（含CPU准备、间隔等待）扣60分钟额度，未突破。CUDA active未单独测量。
|作业|执行终态|科学用途|生命周期秒|
|---|---|---|---:|
|build-attempt01|COMPLETE|有效原始证据|68.907|
|compare-attempt01|COMPLETE|测量INVALID，保留不用|30.544|
|compare-attempt02|COMPLETE|有效原始证据|19.440|
|propagation-attempt01|COMPLETE|有效原始证据|54.404|
|timing-confirm01|FAILED_OR_PARTIAL|被其他GPU进程拦截，0个自有worker|0.067|

全部四个实际spawned owned进程均exit0且PID/birth检查后不存活。额外计时请求启动前遇到文本缓存实验PID163764，未启动、未杀进程、未重试。其他会话作业继续。CPU报告解析曾因未先判断该无结果终态而报FileNotFound，未产生GPU作业/结果；已以JOB原始终态记录。
本地证据约54MiB；远端新实验约22MiB，均远小于2GiB。没有下载/复制整模型。source-final/binary为必要固定证据，不删失败。默认严格版未改。
停止当前数值候选投入：局部快但完整层暂无收益，已观察真实接受翻转，不扩大闭环/训练/阈值或新架构。原预算和deadline保留，未重置其他会话账本。
