# Debug：共享KV与原始torch的逐位对齐

> 2026-10-12T20:01:24+08:00 | zh-CN | Asia/Shanghai

共享KV已经直接对齐原始torch SDPA：复用当前PyTorch2.9对应FlashAttention计算、仅改六处KV读取地址，并移除实验编译中的fast-math。真实3层2,064,384元素、完整30层首个候选Q/K/V与attention输出、全部29个最终候选输出均逐位一致；12类缓存失效、R/HC/S的其他13行隔离、历史/metadata/RNG和原29输出hash检查通过。新S数值门PASS，性能与真实任务闭环尚未验收。

症状是旧S替换原torch FlashAttention后全部29个最终candidate输出逐位失败。先前已排除共享池数据/stride/RNG问题。本轮3个假设：原Triton浮点流程不同（证实并用原Flash math修复）；复用原math后共享取数有误（dense/共享同字节且12失效、13行隔离通过，排除）；独立编译与原wheel的浮点选项不同（fast-math单因素验证，第一版剩1元素，移除后全部逐位通过）。

修复仅在隔离实验中复用原Flash/CUTLASS计算并改六处KV读取，保留CFG2历史及current14、原torch参考、原容差和所有数值/状态/失效/隔离门。三层2,064,384元素与30层/29最终输出均逐位相同。原torch反汇编的非FTZ/精确倒数校正与新编译一致，但不声称相同SASS或唯一某条机器指令因果。

16项CPU测试、48直接attention及161次目标前向、持久闭合计数和GPU清理通过。原失败保留，公开模型和保护状态不改。执行模式为self，代码和独立CPU复查由同一Agent完成，不声称独立Agent复核。

数值阻塞解除；完整性能、E2E和任务成功率尚未验证，无生产启用。本次仅更新研究/Knowledge/PROGRESS与debug日志，不改公共架构。

在相同原torch参考及逐位门下，制定下一轮有界完整verifier性能方案，比较R/S的时间、额外显存和稳定性；采用原先六组交叉顺序，不因数值通过就宣称加速。当前完整性能计划尚未执行，下一计划须携带累计预算，不能把旧585-forward计划直接重跑或默默删门。性能有收益后再考虑task0固定条件的闭环。

详细结果见[完整报告](../research/speculative-sampling/SHARED-PREFIX-TORCH-ALIGNMENT-RESULT-20261012.md)及[CPU审计](../../review/OMISSIONS.md)。
