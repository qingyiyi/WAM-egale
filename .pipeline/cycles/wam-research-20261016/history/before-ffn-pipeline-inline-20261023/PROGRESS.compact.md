# 项目短入口（2026-10-22T22:19:55.013997+08:00）

当前Cycle：wam-research-20261016；固定HW v15语义入口，不重置旧兼容文件。

- M39：当前B FFN profile完成，源码不改。FFN1末组207–209μs，最后FFN2计算92/写回18/等待1μs；local读写29.8MB。NCU的barrier42.5%不是可直接省掉的墙钟。下一步先讨论尾部和额外流量。
- 选定B完整单层warm Graph原Torch/自有短中长1.2185/1.7605/2.8473×；独立FFN仍慢8.9%–9.2%。双方预计算cross KV，不是chunk/E2E。
- 本轮测量副本三KV两进程有限输出/输入/KV/RNG通过，资源一致，完整层扰动≤0.536%；非全输入等价定理。
- 算子累计12603.610/36000秒；当前profile owned已清理。GPU0留算子，不能自动改派机器人。
- M40：用户新批准固定六任务×100初态×T/TF/PF2=1800局，160GPU小时/10天。最新分工GPU1单worker已启动官方有效初态补齐，bank与完整队列自动接续，尚无新质量结果。T0=10/22 22:03:22，10天硬截止11/01 22:03:22，160资源小时更早触发。
- 原RoboTwin DEV90 T25/30、TF24/30、PF223/30；原24h账本54718.244秒及10/22 01:26截止保留。新面板仍扩大DEV，已暴露FIT/CAL/DEV标记不改；不混为独立TEST。
- M17：LIBERO四套6000局齐，严格配对/训练暴露/长尾缺口保留。
- engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission及production_admitted仍false。baseline/投机/原tile/保护state/cycle/rules不改。
- 10/23冻结身份，10/28主表，11/01 18:00中文证据包；证据不足收缩claim。

入口：[Cycle](cycles/wam-research-20261016/PROGRESS.md)、[FFN](experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESULT.md)、[RoboTwin新链](experiments/robotwin-pilot-20261020/eval100-20261022/PLAN.md)、[Memory](memory/INDEX.md)。只按需读取原证据，失败不删，预算不重置。
