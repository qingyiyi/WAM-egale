# 当前讨论与授权（2026-10-22T12:18:20.800737+08:00）

用户最新“那要不先这样吧，你看看能不能先去掉同步，实现tile调度”：当前优先实现依赖就绪调度，替代先完全对齐的顺序。仍保留必要输入/共享内存同步，不把原语慢点全归因grid barrier；不保证异步更快。

已完成B14单层28节点源码、CPU图审查和SM80编译；一次启动grid.sync，按节点完成释放tile，Cross-K/V提前填空。FFN2每输出tile第二段依赖第一段，可能损失原K分段并行。普通节点仍完整输入就绪，cached路径不在本次实现。

原夜间09:00已过。已准备并询问远端GPU0新45分钟窗口，计入原十小时，未获答前不启动。旧累计6004.497秒保留，新GPU0秒；候选DSO未部署。只改optimized，baseline/投机/原tile/保护文件不动；M40旧截止不变。

[结果](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)、[历史讨论](history/before-ready-20261022/DISCUSSION-SUMMARY.md)。
