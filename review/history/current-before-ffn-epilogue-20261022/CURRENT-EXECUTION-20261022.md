# 当前执行：Cross K/V预计算与阶段诊断已收口

自有及强guarded均缓存最终K（含RMS）和V。自有warm短/中/长0.957/1.124/1.140ms，对guarded0.887/1.006/1.021ms仍慢7.94%/11.70%/11.69%；自身比不预计算降低17.7%–20.6%。同分组、不预计算时差距约±2%，不能用这行掩盖最强缓存baseline。

9签名/27结构与三KV缓存baseline有限逐位及状态通过。逐阶段profile与正常性能分开；首次插桩漏记录入口/出口同步的FAILED保留，修复后完成。warm5304/5376tile、31grid；一层一个persistent kernel，无新ready队列。两机研究入口采用，GPU已清理，四科学权限false、旧paused72不改。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)、[完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/STAGE-TABLE.md)、[优化索引](../.pipeline/experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)、[HW](../.pipeline/PROGRESS.compact.md)。LingBot-VA2加速基线核验在上一项分组复用证据目录。

16:53补充：阶段表已增加每tile工作量、尾块与K分段口径；短/中/长warm逻辑任务为5304/5376/5304。FFN2为48个输出tile、96个K分段任务；attention内部KV循环不另算外层任务。此次仅CPU源码核查与文档更新，原GPU计时、源码和科学权限不变。
