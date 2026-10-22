---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-23T02:12:38.420042+08:00'
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
- [ ] M17/M40：M17保留原证据缺口；M40沿原160小时资源临时双卡，GPU0在算子完成后借用至10/23约10点。

[本轮结果](../../experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)。累计18427.875/36000秒；当前作业闭合，原截止与四科学权限不变。

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
| `M40` | RoboTwin迁移与证据 | 新资源授权下固定1800局扩大DEV | 原结果保留，GPU1新链有界运行，GPU0留算子 |

- [x] M39-FFN-PIPE：四组FFN依赖流水完成；R2有限逐位/状态通过，相对此前阶段版整层小幅改善0.33%–0.36%。R1负结果、R2目录错误FAILED及同截止retry保留；owned清理。

- [x] M39-FFN-DIAG：三KV两进程FFN原Torch对照、整层/独立时间线、本机NCU及生产SASS定位完成；R2不改，自有清理，失败保留。

- [x] M39-FFN-TAIL：尾部四块拆为12个M32任务，有限门通过；原Torch三KV完整层1.1910/1.7241/2.7993×，本次延迟降约1.9%。两机研究采用，owned清理。
- [x] M39-FFN-EPILOGUE：A四元素写回、B生产函数内联均有限通过并提速；当前完整层原Torch比值1.2185/1.7605/2.8473×，独立FFN仍慢约9%；两机研究采用，owned清理。
- [x] M39-FFN-REMAINING：当前B时间线＋NCU完成，有限数值/状态通过；已定位末组生产延迟与额外local/指令，不改生产入口。

- [ ] M40-EVAL100：新160GPU小时/10天资源已授权；固定1800局，GPU1已补齐600有效初态与bank，固定质量队列已自动运行；最新本时段GPU0临时跑RoboTwin至约10点。

- [x] M39-FFN-OUTER-ABI：外层流水内联完成，有限门/整层对照通过并采用；后续尾K64也已完成。

- [x] M39-FFN-TAIL-K64：只改12个FFN1尾块K64，有限门及三KV两进程完整层通过，两机采用。
- [x] M39-CURRENT-STAGES：最终当前源码的计算/分块/warp/同步和原Torch阶段比值；GPU0两进程软件插桩测量已闭合，资源属性一致、扰动≤0.462%，原00:58截止保留。

- [x] M39-SELF-QKV-WARP：先核对四warp映射；仅Self Q/K/V固定64×128 tile试8warp，有限精度/状态及完整层原Torch对照后保留或回退。原累计10小时不增加，不恢复旧截止。

Self QKV本轮完成：混合函数负结果保留，隔离8warp有限门及完整层正向，最终全部阶段对原Torch重测、两机采用和清理。下一步同形状其他投影尚未启动。

- [x] M39-PROJECTION-FOLLOWUP：获准接续三个同形状投影及一个有依据Self QKV主循环方案；有限门、完整层与新版完整阶段表，原累计10小时。
- [ ] M40-DUAL-UNTIL10：当前算子结束后，两远端卡去重运行既定完整pair；GPU0按预计任务时长在10/23约10:00自然收尾停派，GPU1继续，总160小时不增。

M39本轮完成，最终同轮原Torch完整单层1.2967/1.8469/2.9910×；GPU0已真实交接，临时双卡首GPU0完整pair审计通过。M40-DUAL-UNTIL10整体待上午自然收尾，不能先标已完成。
