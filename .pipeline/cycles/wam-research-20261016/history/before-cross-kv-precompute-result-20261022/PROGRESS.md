# 当前研究进度（2026-10-22，Cross K/V复用完成）

M39分组复用已完成：9签名/27结构有限逐位与有效状态通过；短/中/长整层1.205/1.372/1.388ms。相比同样分组guarded短快1.69%、中长慢2.08%/2.09%；已有warm缓存guarded0.958/1.078/1.091ms更快。当前还不是跨调用缓存。

[结果](../../experiments/single-layer-async-gemm-20261020/cross-kv-reuse-20261022/RESULT.md)、[阶段表](../../../算子研究/wam_ops/optimized/STAGE-MAP.md)。DSOcc2656e7，普通6200/6272tile、36grid。16:09:21 owned全退出，累计7699.251/36000秒。

用户新批准接入预计算并输出逐阶段耗时，正在CPU实现；新GPU阶段尚未启动。M39-KV-CACHE/M39-KV-STAGES见[PLAN](PLAN.md)，仍先精度后性能，不融合tile与RMS。

M17 LIBERO6000完整局齐，非独立TEST；M40原DEV90完成、旧截止过期，扩大未批准。四科学权限false，baseline/投机/原tile/state/cycle/rules不动。

[此前完整状态](history/before-cross-kv-reuse-result-20261022/PROGRESS.md)。
