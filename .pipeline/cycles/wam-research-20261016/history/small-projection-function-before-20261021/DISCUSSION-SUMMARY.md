# 当前讨论与授权

用户最新同意继续核对并要求解释改了什么、为什么能降到0。本轮没有再次改活动数学实现；核对既有padding136/四元素搬运的边界、地址机制与剩余GEMM成本。0是旧真实独立投影shared写bank conflict，不是耗时或整层全部冲突。

先对齐再调度的决定保持：小投影通用编译路径仍有额外整层成本，FFN自身也未达到Torch；先有限固定形状设备函数对照，再FFN，最后按真实DAG提前无依赖tile。尚未实现ready-DAG，不凭静态local数认定根因。

沿M39十小时累计，当前核对完成与清理；baseline/原tile/投机生产不动。RoboTwin原固定DEV90 GPU1继续，不增加任务/预算/截止，不自动重试。四科学权限false。BAAI仅~/cxz。

[前讨论](history/remaining-gemm-audit-before-20261021/DISCUSSION-SUMMARY.md)。
