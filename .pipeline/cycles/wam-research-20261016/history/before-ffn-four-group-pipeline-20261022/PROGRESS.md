---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T17:44:23.726236+08:00'
current: M39
next: M39已完成融合核验并交付；下一步先定位两处融合的增量开销，再讨论MMA与epilogue流水，不自动开新作业。
---

# 当前研究进度

B14 FFN1+GELU、FFN2+门控残差3已融合并采用为研究入口；9真实签名/36结构条件的有限逐位及状态门通过。两进程同缓存、同真实输入的warm整层Graph计时：相对原Torch短/中/长KV为1.1646/1.6893/2.7445×；比自身融合前慢1.09%/1.14%/0.83%，融合收益为负。仍为阶段同步，尚未实现MMA/epilogue重叠。

warm逻辑任务3400/3472/3400、grid同步29；baseline/投机/原tile不改。旧源码、旧DSO、首CPU编译失败保留；两机采用新DSO，本机未另做GPU计时。四项科学权限及production_admitted均false。

本轮于2026-10-22T17:37:30+08:00清理，原18:12:32截止保留；GPU worker生命周期65.147秒，准备/编译/等待/清理保守记598.644秒，累计8948.557/36000秒。CUDA active未单独测量。无本轮自有GPU进程，不恢复M17/M40旧队列。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | [融合完成但性能退化](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md) | 先分清两处融合开销；新流水尚未实现 |
| `M17` | LIBERO证据 | blocked | 四套6000局已完成；非独立TEST，首两S审计与暴露/长尾缺口保留 | 沿原报告处理，不自动恢复过期作业 |
| `M40` | RoboTwin | blocked | 原DEV90完成；10/22 01:26旧截止过期，1800局/160h未获准 | 保留迁移/优化结果，扩大资源另处理 |

[PLAN](PLAN.md)；[此前完整结果与未完任务](history/before-ffn-epilogue-fusion-20261022/PROGRESS.md)。本次仅交付已授权融合，不以数值门通过替代闭环质量。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。
