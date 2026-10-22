---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T21:26:53.880986+08:00'
current: M39
next: FFN2写回与函数内联已收口；当前独立FFN仍慢约9%，下一步定位当前FFN1/FFN2计算和搬运差距。
---

# 当前研究进度

M39 FFN2四元素写回与FFN1生产函数内联已完成，两机研究入口采用B。A保持数学顺序，减少行循环、重复索引/bias/固定调制读取；B只移除一层生产函数调用，分块与依赖不变。各自9真实/36结构有限逐位、输入/KV/RNG通过；计时fixture与精度一致。四科学权限及production_admitted仍false。

原Torch/完整单层warm Graph，短/中/长为 **1.2185 / 1.7605 / 2.8473×**。A同轮延迟降1.611%/1.349%/1.334%；B在A上同轮再降1.020%/1.156%/0.811%。两独立进程每档均改善，各轮百分比不相加成组合实测。双方预计算Cross K/V，远端A100-SXM4 GPU0；不是chunk或E2E。

选定源码独立FFN重放相对原Torch为 **0.9182 / 0.9167 / 0.9161×**，仍慢 **8.912% / 9.084% / 9.154%**。测量副本完整层扰动绝对值≤0.24%，资源一致、有限输出/输入/KV/RNG通过。旧R2约24%慢是历史测量，不冒充当前结果，也不能宣称FFN已追平。

B的producer独立设备函数消失，合计静态HMMA保留；每线程static local1232→920B，动态local/HBM流量未测。暖数学tile3408/3480/3408、27grid；ACT仍落全局，只有FFN局部流水。baseline/投机/原tile/保护文件不改。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/RESULT.md)、[阶段表](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/STAGE-MAP.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/RESOURCE-AND-STOP-REPORT.json)。本轮GPU worker149.253秒，CPU编译442.124秒，保守含准备/清理1042.947秒，累计 **12338.592/36000秒**。本轮截止22:06:20与旧截止保留；16个owned身份已退出，无本轮失败/timeout；本机只同步版本，未GPU重测。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | 两项有限正向改进；独立FFN仍慢约9% | 当前FFN1/FFN2各自同输入计算/搬运对照；尚未执行 |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；非独立TEST，首两S/暴露/长尾缺口保留 | 不恢复过期作业 |
| `M40` | RoboTwin | blocked | 原DEV90完成；原截止过期，1800局/160h未获准 | 扩大资源独立处理 |

[PLAN](PLAN.md)；[此前状态](history/before-ffn-epilogue-abi-20261022/PROGRESS.md)。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。
