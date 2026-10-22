# 当前：FFN瓶颈定位完成，生产R2保持

独立FFN子图（含GELU和门控残差）相对原Torch：短/中/长KV为0.8080/0.8095/0.8102×，慢23.4%–23.8%。三真实输入、两独立进程，有限逐位/输入/KV/RNG通过。嵌入整层与独立重放的插桩FFN span差不到1%。

72条时间线均显示最后四个FFN1 tile拖后组3；末尾FFN2计算/写回占约68%/25%，提交等待不足约1%。本机NCU诊断路径local读27.54MB/写28.15MB，生产SASS主要显示非内联设备函数寄存器保存恢复；MMA核心是否还有独立瓶颈、各项可消除时间尚未定量。

远端计时与本机硬件计数分开；所有GPU任务已清理，累计10884.364/36000秒。本轮只做诊断，不修改活动源码/DSO，也未实施下一轮调度。PID命名空间误判、Graph未replay先比较、远端NCU权限失败均保留。四项科学权限和production_admitted仍false。

此前R2完整单层相对原Torch1.1646/1.6808/2.7386×是上一轮结果，不是本轮新测整层比值；独立FFN慢并不与整层快矛盾，attention等阶段有不同收益。

下一步优先FFN1尾部工作分摊，再单独验证FFN2写回与函数保存恢复优化；不优先缩小提交等待粒度。

[完整诊断](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)、[时间线](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/FFN-TIMELINE.png)、[状态JSON](CURRENT-EXECUTION-20261022.json)、[HW](../.pipeline/PROGRESS.compact.md)。原paused72交接与RESULTS.json不改。
