---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-22T18:16:14.446638+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前计划：逐阶段优化与证据收口

- [x] M39-MOD：按需调制有限验证与采用，历史结果保留。
- [x] M39-RR：用户批准self RMS+RoPE；实现、编译、9签名/21结构、三KV整层计时完成，两机采用。
- [x] M39-TC：核对投影及attention QK/PV的源码与活动二进制Tensor Core指令；解释RoPE与MegaMoE固定角色。
- [x] M39-RECORD：优化索引、版本身份、失败及资源归档；owned清理。
- [x] M39-KV-AUDIT：按用户新顺序核查Cross K/V分组复用；源码与三档压缩fixture支持每层两CFG组，CPU完成；online RMS文献及当前实现核对完成。
- [x] M39-KV-IMPL：已获准在optimized实现显式context分组，保持独立RMS；CPU审查与编译。
- [x] M39-KV-GATE：真实9签名及结构/条件换序核验，有限逐位通过才进入计时。
- [x] M39-KV-PERF：三KV两进程完整层、强baseline与复用边界；更新阶段表/证据/owned清理。
- [x] M39-VA2：核查LingBot-VA2原文加速基线及推理框架。
- [ ] M39：按最新用户决定，后续仅对比原Torch；历史guarded未追平结果保留，不扩大系统claim。
- [ ] M17/M40：原证据缺口及扩大资源另处理，不启动。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)。累计9844.678/36000秒；当前作业闭合，原截止与四科学权限不变。

- [x] M39-KV-CACHE：用户新批准自有算子读取预计算K（含RMS）/V，双方同等缓存；先精度后性能，冷构建另列。
- [x] M39-KV-STAGES：完整层、逐阶段耗时、同步等待分列；profile与正常计时独立。
- [x] M39-TILE-TABLE：按用户反馈补齐阶段tile数量、每tile形状/工作量、K分段与attention循环口径；仅CPU源码核查及文档同步，不启动GPU。

- [x] M39-GELU-AUDIT：核对GELU 1568tile来源及FFN1/GELU/FFN2/残差依赖；仅源码/旧证据分析，融合与warp重叠未实现、无新计时。

- [x] M39-FFN-FUSE：B14两处写回融合已完成；9签名/36结构有限逐位及状态通过，原Torch三KV两进程整层对照完成。融合本身退化0.83%–1.14%，保留负结果和旧入口；本轮作业已清理。

## 稳定计划ID

M39子项沿用上方名称，以下只补齐语义导航，不重编号历史任务或新增授权。

| ID | 阶段 | 期望结果 | 验证方式 |
| --- | --- | --- | --- |
| `M39` | 单层算子对齐与优化 | 保留有限正确性、性能及负结果 | 当前FFN融合结果；后续先定位融合开销 |
| `M17` | LIBERO证据收口 | 已有四套结果及未通过配对/独立性边界 | 原报告；不恢复过期队列 |
| `M40` | RoboTwin迁移与证据 | 原DEV90及优化证据，扩大预算独立处理 | 原报告与累计资源；不启动1800局 |

- [x] M39-FFN-PIPE：四组FFN依赖流水完成；R2有限逐位/状态通过，相对此前阶段版整层小幅改善0.33%–0.36%。R1负结果、R2目录错误FAILED及同截止retry保留；owned清理。
