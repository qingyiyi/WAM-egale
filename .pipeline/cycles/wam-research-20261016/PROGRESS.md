---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T18:16:14.446638+08:00'
current: M39
next: M39-FFN-PIPE已完成；先汇报有限收益，再评估最后生产组到FFN2的关键串行链与同步成本，不自动开新GPU作业。
---

# 当前研究进度

FFN四组依赖流水R2已完成：FFN1每组28生产tile就绪后派24个FFN2半段任务，两K段MMA并行，第二段仅提交等待第一段BF16部分和。9签名/36结构有限逐位及状态通过；debug三KV均观察组0/1/2提前消费，最后生产和完成均为组3，关键串行链保留。

两独立进程、原Torch同真实输入/最终Cross K/V缓存/warm单层Graph对照：最终1.1646/1.6808/2.7386×；比同轮此前阶段版延迟降0.342%/0.357%/0.332%。幅度很小，不外推应用收益。数学tile3400/3472/3400不变，整层grid29→27，ACT仍落全局；无新专职warp重叠或整层通用DAG。

两机研究入口采用R2，本机未另做GPU测试。R1额外串行化负结果保留；R2首验证因build-lib目录错误在0调用退出，R1原对象重新链接后与原二进制身份一致，R2同一二进制在新retry目录通过。全部FAILED和原截止保留。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)。

2026-10-22T18:10:40+08:00 owned全部退出。GPU worker生命周期136.105秒，含准备/CPU构建/等待/清理保守896.121秒，累计9844.678/36000秒；本轮18:40:44截止不变。四科学权限和production_admitted均false，baseline/投机/原tile及保护文件不改。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | [四组流水有限验证完成](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md) | 评估最后组关键路径，尚未开新作业 |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；非独立TEST，首两S审计及暴露/长尾缺口保留 | 按原报告处理，不恢复过期作业 |
| `M40` | RoboTwin | blocked | 原DEV90完成；10/22 01:26旧截止过期，1800局/160h未获准 | 扩大资源独立处理 |

[PLAN](PLAN.md)；[此前状态](history/before-ffn-four-group-pipeline-20261022/PROGRESS.md)。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。
