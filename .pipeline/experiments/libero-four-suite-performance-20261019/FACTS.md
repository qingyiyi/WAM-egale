# 事实、未知与建议

更新2026-10-20T02:38:05.127898+08:00。

OBSERVED：原四套6000完整终态齐，各臂500；成功、全部失败/长度/罚分见[QUALITY-RESULT](QUALITY-RESULT.md)。八组合max长度比均超旧1.30工程门。M34单层单kernel与warp角色已实测，原精度/速度NO-GO，600调用后收口。新计时301/360，状态RUNNING_PARTIAL_TIMING_PANEL；重复仅10case/init49，不增加质量N，Object少一轮。所有臂启用原等价reset bank。

UNKNOWN：独立TEST/非劣未建立、匹配500局MLP未运行；首两S/组件失败保留。新whole kernel未准入，未测30层/chunk/episode；不存在新系统E2E速度。NCU硬件计数与全量CUDA active UNKNOWN。RoboTwin实际迁移时间/VRAM/吞吐未知；仅查官方源码/权重metadata。

HYPOTHESIS：同步单缓冲、较小GEMM tile、寄存器与粗阶段屏障共同限制新原型；不能从共享padding反事实指定bank conflict唯一根因。PF2改善某些case不等于恢复物理状态，且本面板已有第一次T之前的输入/状态差异。

RECOMMENDATION：保留可用冻结baseline；先处理Q/FFN2/attention的同输入数学契约，再考虑一个成熟SM80 tile流水原型。与已有最强cached/shared-prefix/Graph比较；不把普通缓存写成新贡献。不因开发成功率接近就开放旧质量证书或默认PF2。[日期/止损](DATES-AND-PLAN-20261020.md)。
