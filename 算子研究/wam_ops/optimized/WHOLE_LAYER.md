# 当前单层：分组Cross K/V预计算

正常DSO `0f62944795c64213dfa7106b9707a2f6ec5bd0aa41d9d008e94c2794289a0e67`，一个persistent kernel。显式`prepare_context(context, precompute=True)`在GPU构建每层两组K0/K/V；warm直接读最终K/V，跳过两个投影及K RMS。K0额外保留供审计。默认prepare仍只分组，既有未prepare入口保持兼容。

context_rows独立于self history rows；每个候选仍独立Q16，private KV不共享。条件/权重/device/dtype变更须重新prepare并重新录制Graph，本入口没有自动生产缓存失效管理。

有限9签名/27结构与缓存baseline逐位/有效状态通过；warm0.957/1.124/1.140ms，比强guarded慢7.94%/11.70%/11.69%。普通warm5304/5376tile、31grid；不缓存6200/6272tile、36grid。baseline/投机/原tile及四权限不变。

[完整阶段表](STAGE-MAP.md)、[结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)。逐阶段是独立插桩诊断，与正常层计时分开；首次漏记两处同步的原失败保留，修复后完成。没有新ready调度或GEMM+RMS融合。
