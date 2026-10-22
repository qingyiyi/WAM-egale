# 当前研究进度（2026-10-22T16:37:20.064075+08:00）

M39预计算已收口：有限9签名/27结构/三KV缓存baseline逐位及状态通过。自有warm短/中/长0.957/1.124/1.140ms，对强guarded0.887/1.006/1.021ms仍慢7.94%/11.70%/11.69%；自有自身降17.7%–20.6%。不预计算时差异−1.79%/+1.82%/+2.02%。双方同缓存后差距更大，但绝对成本都降低。

[结果](../../experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)、[完整逐阶段表](../../experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/STAGE-TABLE.md)、[优化索引](../../experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。当前warm5304/5376tile、31grid；源码与旧分组路径保留。诊断漏计入口/出口同步的原FAILED保留，修复版在同窗口完成；计算代码和原门未改变。

2026-10-22T16:36:10.992744+08:00owned全部退出，累计8349.913/36000秒。四科学权限false；baseline/投机/原tile/保护文件不动。下一步按最大剩余阶段对齐，不自动恢复ready或开启机器人。

M17四套LIBERO6000局齐，非独立TEST，暴露/首两S/长尾缺口保留。M40原DEV90完成、10/22 01:26旧截止过期，1800局/160h未获准。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。

[PLAN](PLAN.md)、[此前状态](history/before-cross-kv-precompute-result-20261022/PROGRESS.md)。
