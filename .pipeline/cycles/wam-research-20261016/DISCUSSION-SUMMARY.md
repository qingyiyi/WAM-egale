# 当前讨论（2026-10-22T18:59:32.671003+08:00）

用户批准先定位FFN。本轮保持生产R2及调度不变，生成测量副本，做三真实B14输入×两进程独立FFN原Torch对照及整层内/独立时间线；嵌入差不到1%，独立FFN慢约24%。FFN1尾部四tile集中组3，最后一组FFN2主要为计算和写回，提交等待约1us，故不优先细化提交等待。

本机NCU补到显著local请求（诊断同计算路径），生产SASS表明主要是非内联设备函数保存恢复；未证明该访问贡献全部时间差，也不说MMA内频繁spill。FFN2最终epilogue还有通用索引/门控开销，尚未单因素归因。下一步建议FFN1尾部分摊，再单独优化写回和ABI开销，不能仅放大所有tile或直接恢复大DAG。

PID命名空间误比、Graph捕获未replay先比较、远端NCU权限失败均保留；前两项修脚本，同二进制完成，计数在本机补采且不混性能。自有任务已清理，累计10884.364/36000秒；原截止和四科学权限不变。没有新增生产优化。

[诊断结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)、[此前讨论](history/before-ffn-diagnosis-20261022/DISCUSSION-SUMMARY.md)。
