# Cross K/V 与 FFN 设备函数控制：两项完成，保留修改

2026-10-21，M39。结论：固定数学、tile 和阶段同步，仅改变设备函数组织，B14 完整单层相对本轮修改前下降 **9.46%–10.87%**；仍比 native 慢 **15.28%–30.53%**。本轮没有实现跨阶段 ready 调度。

## OBSERVED：完整层直接对照

以下均为真实同一输入、同一进程交错测得的完整层 CUDA-event 延迟；每格固定60次预热、12次测量取中位数。两次独立进程都保留。最终对照直接加载本轮最初二进制与最终二进制，没有将分轮收益相乘。

|KV|进程|native ms|本轮前 ms|最终 ms|相对本轮前下降|最终比native慢|
|---|---:|---:|---:|---:|---:|---:|
|288|1|2.119680|3.104256|2.766848|10.87%|30.53%|
|288|2|2.120704|3.102720|2.766336|10.84%|30.44%|
|1152|1|2.405152|3.367968|3.026832|10.13%|25.85%|
|1152|2|2.405088|3.364512|3.028480|9.99%|25.92%|
|2160|1|2.846080|3.628832|3.280896|9.59%|15.28%|
|2160|2|2.844112|3.623584|3.280912|9.46%|15.36%|

原件：[进程1](profile/TIMING-1.json)、[进程2](profile/TIMING-2.json)、[三轮全部18格](FULL-LAYER-ALL.csv)。`profile/`只是证据目录名；上述正式对照样本使用未插桩二进制，profiler在样本之后单独运行。

条件：BAAI A100-SXM4-40GB GPU0、CPU4/5，GPU0在各轮只有本作业；GPU1同时运行RoboTwin，整机不是独占。固定预热后观测SM1410/MEM1215 MHz，没有改时钟。Torch2.9/cu126、私有CUDA12.4、SM80、BF16。计时包含完整单层调用及其workspace分配/提交，准备权重与安装fixture在外；这是uncached/eager层级诊断，不是最强cached/Graph、30层verifier、chunk或机器人E2E，也不解除正式性能权限。

## OBSERVED：改了什么

只改活动 [whole_layer.cu](../../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)，baseline、原tile和投机生产代码未改。

1. `context_linear`（171行）：整个Cross K/V阶段进入一次专用设备函数，tile循环内部直接展开固定CUTLASS计算体。原来是`linear → 每tile调用通用gemm_tile`。N/K固定3072，实测context M1024或7168，tile仍128×128、K步长64，没有新增BF16分段。
2. `action_ffn_linear`（185行）：只对B14 M224固定FFN形状。FFN1仍128×128、K3072；FFN2仍64×128、K14336，保留两个K7168分段和原BF16/bias边界。B2 action/video FFN仍走原通用路径。

调用位置611/612/622/627行。CUTLASS核心、queue、tile完成同步与阶段grid.sync均未改。仍为一个cooperative kernel、108常驻CTA、256线程；不是把7候选当Q112，也不是跨阶段warp任务领取。

独立增量：只改Cross时整层下降8.84%–10.60%；在Cross版本上再改FFN下降0.587%–0.700%，约17.4–22.4µs。FFN收益小但六个配对方向一致，因此保留；不将它夸大为已经对齐FFN。

## OBSERVED：阶段profile与扰动

外部生成两份诊断二进制，在同一阶段开始/结束边界记录globaltimer。没有将标记加入活动源码。以下为8次样本中位数，µs：

|阶段|短KV 前→后|中KV 前→后|长KV 前→后|
|---|---:|---:|---:|
|Self Q（未针对修改）|51.200→51.712|51.712→51.712|52.224→52.224|
|Cross K|792.064→609.280|794.624→609.792|796.160→610.816|
|Cross V|787.456→606.720|785.408→607.744|785.408→607.232|
|FFN1|237.056→215.040|235.520→214.016|235.520→213.504|
|FFN2|166.912→158.720|167.936→159.744|166.912→159.232|

[阶段原件](profile/RESULT.json)、[阶段CSV](GEMM-PHASES.csv)、[插桩控制](INSTRUMENTATION-CONTROL.csv)。插桩前版本整体约+0.47%至+1.00%，后版本约−0.88%至−0.94%，且静态local大小改变。故该表用于方向和量级诊断，不能把阶段差额相加当成未插桩总收益；完整层收益以上表独立未插桩计时为准。本轮只重测这五个GEMM阶段，旧全29阶段表仍是旧版本证据。

## 机制：证据支持什么

旧通用`gemm_tile`有多个shape路径，并在函数入口/出口存在大量STL/LDL寄存器保存/恢复。新`context_linear`把这些入口/出口工作放在整个tile循环外；循环内直接包含CUTLASS设备代码。[反汇编边界摘录](SASS-FUNCTION-BOUNDARIES.json)、[旧统计](cross/before/SASS-SUMMARY.json)、[Cross统计](cross/tested/SASS-SUMMARY.json)、[最终统计](ffn/tested/SASS-SUMMARY.json)可追溯。

按源码逻辑，B14每个Cross投影1344个tile：旧通用设备函数每tile调用，新阶段函数每CTA进入一次（108个CTA）。平均每CTA约12.44个tile，函数入口成本有更多摊薄空间。FFN1只有224个tile，FFN2只有96个tile，摊薄空间更少。这里是源代码推导，不是NCU动态调用计数。

控制实验支持“设备函数组织/编译结果是Cross额外成本的重要来源”。不能单独量化其中函数调用、寄存器保存/恢复、常量传播和分支消除各占多少。静态local只由1472降至1456B/线程，register仍255、shared仍98848B、静态可驻留仍1CTA/SM；不能把更快误写成occupancy提高。

**UNKNOWN**：动态spill、L2/HBM流量、实际occupancy、各类保存/恢复的动态时间。没有新增NCU硬件counter数据，不能声称HBM流量或spill下降某个百分比。STL/LDL包括函数ABI保存，不等同全部是寄存器分配器spill。静态指令数还包含互斥路径，不能直接充当执行指令数。

## OBSERVED：有限精度与单kernel

Cross和FFN两阶段各自完成9个真实签名：B2 action、B14 action、B2 video × 短/中/长KV。每签名23个可配对中间量，最终输出与Torch逐位相同（max_abs=0）；共18个有效KV状态转移、输入/RNG及重复结果检查通过。B14只读有效KV也单独核对。

[Cross有限门](cross/SIMPLE-GATE-RESULT.json)、[FFN有限门](ffn/SIMPLE-GATE-RESULT.json)。最终两进程三KV再核对输出/中间量/有效KV；分别单独profiler观察到一个`whole_layer` kernel。

这是固定环境和有限fixture证据，不证明所有输入、全模型scheduler/接受路径或机器人闭环质量。四项质量/独立TEST/非劣/正式性能权限保持false；旧FAILED、retention和paused72不变。

## 资源与终态

三次有界作业都COMPLETE_ATTEMPT，未发生新GPU失败/超时。两候选，4869次包含参考臂的完整层调用。GPU worker生命周期120.995秒；CPU准备/编译523.226秒；作业墙钟合计644.734秒，首启动到末退出1024.725秒，不能把这几个口径混称GPU计算时间。CUDA active时间未单独测量。

M39累计1770.668/36000秒，原10h预算未重置；本轮≤1800秒GPU、≤2400秒CPU准备和原14:00:42截止均遵守。12:19核验所有worker PID+出生时间已退出，GPU0无compute进程；GPU1 RoboTwin继续。证据目录约153.4MiB，低于1GiB。本轮没有以失败重试追加样本。

[完整账本](RESOURCE-AND-STOP-REPORT.json)、[清理核验](CLEANUP-VERIFICATION.json)、[最终选择](FINAL-SELECTION.json)、[原登记](PLAN.json)。活动源SHA256 `0b7779635bfd79fc163abe44acd7682c92196d49ccef6d0527ddd5f4ce300aee`；原件/诊断源见[manifest](profile/SOURCE-MANIFEST.json)。完整DSO、SASS与profiler原件保留远端，公开仓库只同步小证据。

## RECOMMENDATION：下一步

保留本轮两项修改，停止当前候选搜索。先继续核对FFN原语中的MMA主循环、两段归约和epilogue成本，固定数学边界逐项比较；不能把此前独立M128并行K接近Torch的结果直接集成，因为其整层负结果仍有效。

Cross大幅额外开销已缩小，但完整层仍慢15%–31%，FFN独立原语仍有差距。因此下一轮应先处理剩余原语/资源问题，再设计以cross K/V为可提前分支的最小DAG调度。每子算子按合适tile，不凑108，不用全局等待总和预测加速。本轮没有为调度提前动代码。
