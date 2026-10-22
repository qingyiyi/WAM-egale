# 当前讨论与授权（2026-10-22T07:59:51.237516+08:00）

用户同意继续单层性能对齐至10/22 09:00，原累计10 GPU小时。先对齐，再tile ready调度；不能完全对齐需给测量和合理原因。只改optimized，baseline/原tile/投机生产不动，BAAI仅~/cxz。

本段已终态：两项设备源码对照无稳定收益；编译链新证据触发前瞻第三项LTO，精度通过却慢约16%，全部恢复11。LTO的Cross-K/V只有嵌入整层明显退化，FFN1/2改善；不能把全部慢因说成同步或认为localSize更小必然更快。当前版仍慢于同工作量uncached guarded约11%–13%。

下一轮建议限定FFN设备编译边界，同时保留Cross-K/V现有快路径；B14专用入口后备。未实现这两项，不继续无边界搜索。cached命中时Cross-K/V不是可领取任务，不能把无效计算当DAG补位收益。当前无ready-DAG。

本机/BAAI本段owned都已退出，原截止/预算未重置。M40原01:26截止已过，没有1800局/新160h批准；当前算子继续授权不转成机器人资源。四科学权限false；原FAILED/retention/paused72和保护文件不变。

[此前讨论](history/morning-before-20261022/DISCUSSION-SUMMARY.md)。
