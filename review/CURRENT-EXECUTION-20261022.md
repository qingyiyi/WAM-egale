# 当前状态：2026-10-22T22:26:28.905006+08:00

当前B源码/DSO不改，FFN profile完成。FFN1末组207–209μs就绪，末FFN2计算约92μs/写回18μs/提交等待1μs；NCU local读写29.8MB。当前SASS显示绝大多数local指令在FFN流水函数入口/出口，支持保存恢复来源，但尚未做移除边界的因果对照。barrier42.5%不是可省墙钟。

既有选定B完整单层原Torch/自有1.2185/1.7605/2.8473×；独立FFN仍慢约9%。本次是profile，不是新性能优化或闭环质量放行。[详细结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESULT.md)、[时间线](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/FFN-TIMELINE.png)。

RoboTwin新160GPU小时/10天已获准；最新固定GPU0留算子、GPU1单worker跑1800局。GPU1正在补齐每任务19→100有效初态，后接CPU核验/full512 BF16 bank/完整T-C-P队列。尚无新完整质量结果；旧DEV90与54718.244秒/原截止保留。新面板仍扩大DEV，旧FIT/CAL/DEV暴露必须标记。[新队列计划](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/PLAN.md)。

所有科学权限仍false，review/RESULTS.json历史暂停72局不改。源码、baseline、投机实现与保护文件均未改。当前M39累计12603.610/36000秒。
