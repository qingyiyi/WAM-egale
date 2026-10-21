# 当前执行证据（2026-10-21T14:41:26.005701+08:00）

当前单层数学源码仍为df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a。此前完整uncached/eager B14两进程仍慢：短29%、中25%、长14%；FFN1尾块仅改善0.19%–0.50%。本轮没有新增优化候选。本机NCU已完成，24秒后GPU释放：三对GEMM的Tensor活跃更低且出现local读写，FFN1 DRAM为原版2.083倍；这是一轮hot诊断，不是正式性能或唯一根因证明。

三KV阶段/CTA诊断完成420完整调用与10962阶段调用，有限逐位/有效KV/输入/RNG通过，插桩扰动约1%。剩余耗时优先定位Cross K/V、小action投影、FFN、Cross K RMS和GELU；屏障最后到齐后的释放中位约1µs，早到CTA等待混有其他CTA仍在计算，不能全称纯同步。native外侧marker与自有函数调用点边界不同；这是诊断定位，不是E2E加速表。

- [完整慢因分析](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)、[三KV时间线](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/TIMELINE-RESULT.md)
- [阶段明细](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/STAGE-SUMMARY.csv)、[CTA等待汇总](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/BARRIER-SUMMARY.csv)、[有限逐位与插桩扰动原JSON](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/trace-r2/TRACE-RESULT.json)
- [资源与失败](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESOURCE-SNAPSHOT.json)、[测量wrapper修正记录](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/HARNESS-CORRECTIONS.md)
- [本机NCU结果](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/local-ncu-exclusive/NCU-SUMMARY.json)、[终态](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/local-ncu-exclusive/FINAL-STATUS.json)，以当前文件为准；远端NCU权限拒绝记录保留，不能以静态local字节声称动态spill已证实。

RoboTwin固定DEV90当前快照完整16/90，状态RUNNING_COMPLETE_DEV_EPISODE，当前open_microwave-valid14-T；[STATUS](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)。初态与C/P首两S审计、原失败、24h累计和10/22 01:26截止保持。GPU1后台自动接续既定完整队列，不自动重试。

四科学权限false；LIBERO及DEV非独立TEST。baseline、原tile和投机生产不改。原始CTA时间线/输入tensor/权重留本地；公开派生统计与源身份，不声称网页端可独立重算所有张量。旧paused72及RESULTS.json不变。
