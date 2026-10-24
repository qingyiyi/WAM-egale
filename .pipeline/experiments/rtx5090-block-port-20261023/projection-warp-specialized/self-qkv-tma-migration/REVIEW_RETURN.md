# Self Q/K/V局部warp流水：有限结果

## 结论：研究采用

OBSERVED：当前采用SELF-QKV-TMA-4MMA。原Torch/新版完整层短中长为 **1.2475/1.7553/2.1872×**；相比同轮冻结旧A，完整层延迟下降 **2.942%/2.808%/2.249%**。两个独立进程六个shape×process均改善（2.208%–3.028%）。Guarded/新版 **1.1021/1.0850/1.0433×**；四臂均eager、无CUDA Graph，同样预计算Cross K/V。不是机器人E2E结果。

## 实际改动

Self Q/K/V共288个输出任务和64×128 tile保持；由原八MMA warp改成W0 TMA、W1–3 epilogue、W4–7 MMA。每MMA warp负责32×64输出；输入保留双K64槽。十次数值分段9×K320+K192、段内K16次序、BF16归并和最后bias顺序不变。只扩充六组投影descriptor/调试计数，Self Q/K/V读相同M1但权重、输出仍独立。不新增跨候选attention、不合并QKV权重、不改FFN或外部stage grid同步。完整diff为`CHANGE.patch`。

## 正确性与计时证据

- 三份真实B14/Q16、block0、KV288/1152/2160输入：27个捕获中间量/输出、normal/debug、输入、全部有效KV和CPU/CUDA RNG有限逐位通过；保存误差均0。独立候选接口先完成三KV精度，再执行性能。
- 原Torch、Guarded、旧A、新版：两独立进程，各shape12轮预热、16轮交错×5完整block；前后输出逐位通过。正常完整层决定采用，阶段来自单独插桩版本。
- 实际公开loader默认流及非默认流各三KV通过；加载module及binary路径核对。六组投影每tile计数均48次A、48次B、10次epilogue、48次MMA输入消费；各warp角色计数通过。FFN原672任务/有序提交计数仍通过。
- Self QKV阶段原Torch/新版为 **1.339/1.316/1.315×**；其他阶段及完整融合FFN见`STAGE-TABLE.md`。插桩完整层扰动短中长为-0.02%/+0.23%/+0.04%，仍不是纯GEMM吞吐或精确硬件stall分解。
- 正常静态资源仍为`[170, 1, 100352, 255, 784, 256]`：170 CTA、每SM1 CTA、100352B shared、255寄存器/thread、784B localSize/thread、256线程/CTA。没有把静态localSize等同实测访存流量。

## 未知与后续

UNKNOWN：未访问硬件计数器，不能量化TMA与MMA重叠比例、真实HBM/L2带宽或寄存器依赖stall；未重验其他层/模型/在线接受路径/机器人闭环。四科学门和production_admitted保持false。

HYPOTHESIS：288个Self QKV任务比单独96任务更易覆盖170常驻CTA；本轮没有单独控制所有队列/分工因素，不能只用该解释证明机制或外推其他shape。

RECOMMENDATION：先保留本次已测收益；后续优先针对仍慢于Torch的融合FFN和三个96任务投影分别做有限归因，不统一扩大warp或tile、不提前上完整DAG。

## 资源与归档

本job生命周期（含两个构建、等待、公开入口核验和最终回传）434.383秒，保守计480秒；完整block 2871/12000次，两个构建和五个GPU进程均成功、无timeout。CUDA active时间UNKNOWN，不能用生命周期代替。原十小时账本35926.41251785547秒保留；本次按用户新5090追加授权登记，不改A100预算。

自有PID+出生身份及进程组已退出，5090 compute context为空，实例未关机。源码/二进制六文件回本机byte一致，无SHA。源码、逐次计时、负结果/历史A和本轮状态均保存；未改baseline、投机代码、protected state/cycle/rules。

固定入口：`SELECTED.json`、`SELECTED-source/`、`SOURCE-AND-RESULT-ENTRY.json`；原始计时`results/measure1-data/RESULT.json`和`results/measure2-data/RESULT.json`；资源`RESOURCE-AND-STOP-REPORT.json`；全表`STAGE-TABLE.md`及`snapshots/FINAL-STAGE-TABLE.md`。
