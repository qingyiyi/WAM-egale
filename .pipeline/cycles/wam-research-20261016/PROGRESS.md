# 当前研究进度（2026-10-22T12:46:50.150956+08:00）

|ID|状态|证据与下一步|
|---|---|---|
|M39|in_progress|ready有限精度通过、性能NO-GO；用户要求回退阶段同步并重新顶层规划|
|M17|in_progress|LIBERO6000局齐，非独立TEST，暴露/首两S/长尾缺口|
|M40|in_progress|工程4局与旧DEV90已归档；旧截止过期，扩大未批准|

## M39最新结果与当前源码

新ready路径9签名/81调用、21结构/84调用有限逐位与状态通过；两进程Graph完整层短/中/长3.765/3.887/3.858ms，旧11为2.292/2.457/2.477ms，回退64.22%/58.20%/55.72%。同构建顺序入口仅偏差−0.134%至+0.115%，回退集中在ready执行路径。

[完整结果](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)。CTA累计领取约21%、发布约4%，不是墙钟比例；插桩扰动0.46%–1.21%。去掉grid等待没有带来整层收益。不能把已知queue开销说成唯一根因。

按用户新决定，optimized源码恢复阶段同步旧11、默认DSO始终ac3f9e46；ready源码与失败/负结果保留在证据ready-source。当前先讲清完整链公式、tile和39处grid同步，重新规划任务粒度与合并边界，不继续直接优化细ready队列。baseline、投机、原tile不动。

12:28:18起新45分钟获准，12:43:09清理闭合，GPU worker生命周期125.117秒；整窗含CPU/等待/清理保守计费891.311秒，累计6895.808/36000秒；旧09:00和新13:13:18截止均保留，不据余额自动开新作业。CUDA active未测，所有owned退出。

## 保留的主线结果

LIBERO各方法4×10×50=2000，共6000局。T/TF/PF2成功数：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499（各500）。非三seed、非独立TEST，曝光/首两S/长尾缺口保留。

RoboTwin旧DEV90：T/C/P25/24/23（各30）；跨任务工程4局已终态，不新增独立质量N。累计54718.244/86400秒、10/22 01:26:04截止不变，1800局/160h未启动/未批准。

四项科学权限false；旧FAILED/retention/paused72与state/cycle/rules不动。BAAI仅~/cxz。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。

[本轮前状态](history/before-ready-gpu-result-20261022/PROGRESS.md)、[晨间结果](../../experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)。
