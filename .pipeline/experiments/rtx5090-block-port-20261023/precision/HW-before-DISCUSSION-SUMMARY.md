# 当前讨论（2026-10-23T19:38:39.747829+08:00）

5090完整单层已实际迁移，终态CLOSED_DIAGNOSTIC_NOT_QUALIFIED。B14真实短/中/长KV288/1152/2160，两个独立进程，正常二进制双方eager原Torch比值1.2201/1.7155/2.1406×，双方Graph1.1062/1.6024/2.0361×。输出相对L2为0.9387%/0.7519%/0.6759%，最大绝对差2.5/2/2，首次分歧在Self QKV；不能称精度对齐加速。LN1/调制1、有效KV、输入/RNG门通过，仅三fixture，不继承A100证书。初轮compact-input缓存不等价；最终改为原B14输出去重，Torch缓存baseline恢复逐位一致，自有仍失败。原r1/4次构建失败保留。

用户要求迁移5090测试性能和完整阶段表，已实际执行。结论是可运行、可测量但未通过数值门，当前仅保留独立SM120研究副本。下一步优先投影数值契约与FFN，不以小相对L2自动放行、不用慢baseline制造PASS。利用率异常本轮结束显示0%但原因未知，不为此改驱动；counter仍受宿主管理员限制。

[结果](../../experiments/rtx5090-block-port-20261023/REVIEW_RETURN.md)、[本次更新前讨论](../../experiments/rtx5090-block-port-20261023/PRIOR-CYCLE-DISCUSSION-SUMMARY.md)。原RoboTwin队列与截止不变，未在本轮重新巡检。
