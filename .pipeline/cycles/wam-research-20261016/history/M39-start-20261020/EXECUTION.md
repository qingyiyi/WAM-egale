# 当前Execution checkpoint

## 2026-10-20T09:57:08.243204+08:00：M38有界profile完成

用户最新优先解释当前整层算子慢点，M38新登记2700秒/240block/600原语/512MiB。实际86.490秒含57.551秒CPU构建、28.716秒GPU worker；222block/300原语、六真实签名/144完整计时/18trace。六原Torch/native逐位通过、whole均失败，profile/static六输出/状态匹配失败v1。

whole比native慢5.09–11.14倍；插桩GEMM61%–84%，同输入库GEMM差约一个量级；静态领取中位数慢3.1%–12.8%，并非去队列即解决。实际compute kernel与memory/annotation按Chrome类别纠正分开，原始数据不改。阶段与完整计时分离，NCU硬件计数UNKNOWN。

[结果](../../experiments/whole-layer-bottleneck-20261020/RESULT.md)、[清理](../../experiments/whole-layer-bottleneck-20261020/RESOURCE-AND-STOP-REPORT.md)。parent/build/worker PID与birth核对均退出、两卡0MiB；无其他排队作业。baseline/主whole/投机源与全部旧FAIL/四false保持。下一建议M39，未启动。

之前M37与本轮起点原文见[归档](history/M38-start-20261020/EXECUTION.md)。
