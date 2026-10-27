# 资源与收口

T0=2026-10-27T09:42:41.496368+08:00；终态=2026-10-27T10:10:36.436233+08:00；原绝对截止12:42:41不变。

|作业|生命周期秒|终态|PID / 出生ticks|清理|
|---|---:|---|---|---|
|build-attempt01|69.511|COMPLETE|169459 / 2650374491|owned_alive_after=false|
|layer-attempt01|35.540|COMPLETE|169559 / 2650389253|owned_alive_after=false|
|policy-attempt01|384.474|COMPLETE|170740 / 2650445168|owned_alive_after=false|

实际runner累计489.526秒（8.16分钟）。从登记起含准备、等待、运行和归档保守扣账1674.9秒（27.9分钟）≤5400秒；本轮剩余额度不再使用。CUDA active未独立测量，不用runner秒冒充active。

独占检查无其他GPU context，三个自有worker均自然退出，终末nvidia-smi无compute进程。未结束租机、改系统服务、杀他人任务或重启三A100作业。

本轮无GPU失败/timeout，未触发实现修订。一次本地rsync目的父目录不存在的归档错误修正后重新复制；不涉及GPU重跑或科学数据修改。新证据本地约18.9MiB，远端目录体积以终末df/du收据为准，≤2GiB；未复制模型权重/全历史KV。

停止：未扩大至新suite或闭环，不以compile路径相同代替新版质量；缓存作为共同工程优化，不申报为调度创新。四科学权限仍false。严格FFN2不变；生产Torch与全局默认不变。
