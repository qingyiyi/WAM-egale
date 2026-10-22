---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T19:53:45.599095+08:00'
current: M39
next: FFN1尾部分摊已收口；按已确认顺序继续FFN2写回与函数开销，不扩大机器人任务。
---

# 当前研究进度

M39 FFN1尾部行块分摊已完成并用于两机研究入口。112个生产tile改为108个M128大块+12个M32尾块，共120个；FFN2的96个任务与原归约/舍入不改。9真实签名、36结构控制有限逐位/输入/KV/RNG通过，计时和精度fixture一致。四科学权限及production_admitted仍false。

两独立进程、三KV，远端A100-SXM4 GPU0无争用、双方预计算Cross K/V、完整单层warm Graph相对原Torch为 **1.1910 / 1.7241 / 2.7993×**。本次相对同轮R2内部回归控制，完整层延迟降 **1.932% / 1.918% / 1.856%**。这是单层，不是chunk/WAM/E2E。

debug最后组就绪相对首组的间隔约118–119→89–90us，最后消费完成相对首组约246→224–225us；支持尾部分摊有效。最后生产后的消费尾段反而127→134–136us，不能称FFN2已优化。static local1144→1224B，动态流量未重测。旧独立FFN慢约24%是旧R2结果，不能挪作当前新结果。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/RESULT.md)、[阶段表](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/STAGE-MAP.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/RESOURCE-AND-STOP-REPORT.json)。本轮GPU worker59.630秒，CPU编译147.450秒；保守含准备/清理411.282秒，累计 **11295.646/36000秒**。新截止20:29:24和旧截止保留，owned已退出，无本轮失败/timeout。本机只同步源码/DSO，未做GPU重测。

warm数学tile3408/3480/3408、27grid；ACT仍落全局，只有FFN局部流水。baseline/投机/原tile/保护文件不改。旧NCU/诊断、FAILED、R2源码/DSO均保留。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | FFN1尾部分摊有限通过，约1.9%整层增量 | FFN2最终写回/索引，随后函数保存恢复；尚未执行 |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；非独立TEST，首两S/暴露/长尾缺口保留 | 不恢复过期作业 |
| `M40` | RoboTwin | blocked | 原DEV90完成；原截止过期，1800局/160h未获准 | 扩大资源独立处理 |

[PLAN](PLAN.md)；[此前状态](history/before-ffn-tail-20261022/PROGRESS.md)。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。
