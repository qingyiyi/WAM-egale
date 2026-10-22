# 事实、未知与建议

OBSERVED：三新候选均精度有限通过，前两无稳定收益，全局LTO慢15.6%–16.2%，全部撤回。当前11未完全对齐同工作量强baseline。当前版4范围NCU完成；LTO逐阶段诊断完成；全部owned清理。

UNKNOWN：当前版local具体动态热点、LTO退化的精确spill/缓存归因、全部在线输入等价、cached引擎下新调度收益。

RECOMMENDATION：保留11，下一轮先限定FFN编译边界并保留Cross-K/V快速路径，暂不做ready-DAG。[完整证据](RESULT.md)。
