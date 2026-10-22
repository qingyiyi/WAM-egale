# 下一步：保留测得更快的版本，缩小对齐问题

OBSERVED：本轮三项对照均不晋级；生产研究代码恢复selected11。完整层约2.30/2.46/2.48ms，仍比同工作量uncached guarded慢约11%–13%。不把LTO选项缺失直接当成已找到的性能根因，它实际更慢。

优先完成已有逐阶段诊断归因；本轮不继续参数/编译网格，不实现ready-DAG。LTO逐阶段新证据后，下一轮首选保留Cross-K/V现有快编译路径，只限定FFN的设备函数编译边界；可研究混合IR/普通设备对象链接，仍只有一个kernel。这尚未实现/未保证收益。B14专用入口缩短活跃状态作为后备，不同时展开。原本现场逐位门、三KV及尾部/CFG控制必须保留。不能仅看静态register/local或SASS长度判优。

进入调度研究前还须明确缓存口径。`算子研究/wam_ops/optimized/prepared.py:64`的fill使用原形状Linear缓存Cross-K/V；`:109`的activate(cached=True)复用结果。本轮五臂完整层计时明确cached=False。现在的“强baseline”是同工作量uncached强baseline，不是已战胜全部cached路径。

用户提出的“尾波不满108时提前领取无依赖Cross-K/V tile”在uncached路径有真实候选任务；若条件缓存命中，这些投影不再执行，不能作为免费补位任务，也不能把多做这些投影算成提高利用率。后续必须按cold/uncached与warm/cached分别列DAG和成本，再选择实际存在的独立tile。候选私有KV仍不共享。

当前没有ready-DAG实现、没有30层融合/在线替换；四项科学权限false。RoboTwin原截止已过，不以本次算子继续授权启动1800局或新预算。
