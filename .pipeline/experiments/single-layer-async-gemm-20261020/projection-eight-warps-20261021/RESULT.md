# Cross K/V投影8warp实验：两种方案均未改善完整层

核对时间：2026-10-21T15:46:22.848347+08:00。M39现有工作内直接试验；两机活动源码已恢复原4warp，远端DSO也已恢复。

## OBSERVED

原投影由W3–W6执行CUTLASS cp.async/MMA；W1/W2在这条路径没有独立搬运工作，W7主要等到共同epilogue，W0仅thread0在tile边界领任务。它们并非整个block都等待：GEMM写回与若干逐元素阶段使用全部8warp。这个分工从旧小M专用流水沿用，不能仅凭等待warp数量推算时间损失。

本轮仅修改Cross K/V的大M投影，FFN、self/cross attention和action小投影数学路径未主动改动；但整层共享资源及编译布局可能受影响。没有跨阶段ready调度。

|方案|CTA输出tile|MMA warp tile|计算warp|动态shared bytes|寄存器/线程|local bytes/线程|resident CTA/SM|
|---|---|---|---|---:|---:|---:|---:|
|选中原版|128×128|64×64|W3–W6|98848|255|1456|1|
|大tile对照|256×128|64×64|W0–W7|148000|255|1464|1|
|固定tile对照|128×128|32×64|W0–W7|98848|255|1424|1|

8warp两版均按4个M分块×2个N分块映射，W0–W3对应前64列、W4–W7对应后64列。named barrier随实际计算线程改为256；原4warp路径仍为128。加载和计算由同一组warp执行；没有新增专职producer/scheduler warp。K64三级流水、K累加顺序及BF16/bias边界保持。

B14 context矩阵M7168、N/K3072：原/固定tile为1344任务，大tile为672任务，persistent worker池仍108CTA；候选数和独立Q16语义未改变。

## 同轮完整单层计时

远端A100 GPU0无其他计算进程，GPU1 RoboTwin后台继续；uncached/eager完整单层GPU event，准备/安装fixture在计时外。每臂固定60轮warmup，12次交错样本，两个独立进程。表内为各进程12次中位数，不合并挑优。各采样点SM1410MHz、memory1215MHz；未修改时钟。不是机器人E2E，也不是质量放行后的正式性能。

|方案|进程|KV|native ms|原4warp ms|8warp ms|相对4warp变慢|
|---|---:|---|---:|---:|---:|---:|
|large256|1|short|2.115072|2.737664|2.824192|3.161%|
|large256|1|middle|2.402704|2.999712|3.114464|3.825%|
|large256|1|long|2.840016|3.251360|3.408928|4.846%|
|large256|2|short|2.122752|2.738688|2.825728|3.178%|
|large256|2|middle|2.402224|3.003648|3.116784|3.767%|
|large256|2|long|2.842032|3.255888|3.410912|4.761%|
|fixed128|1|short|2.117120|2.736640|2.772992|1.328%|
|fixed128|1|middle|2.404240|3.000224|3.034736|1.150%|
|fixed128|1|long|2.841344|3.247440|3.283888|1.122%|
|fixed128|2|short|2.118144|2.737664|2.774016|1.328%|
|fixed128|2|middle|2.403168|3.000624|3.032112|1.049%|
|fixed128|2|long|2.841728|3.250336|3.290480|1.235%|

大tile版本完整层慢3.16%–4.85%；固定tile版本慢1.05%–1.33%。全部12个方案/进程/KV比较均退化，没有稳定收益，两个候选均不选用。

## 有限正确性与范围

两候选各9真实签名（B2 action/B14 action/B2 video × 三KV），每个23个中间量、输出与Torch/native有限逐位通过，max_abs=0；全部有效KV、repeat、B2三种合法cache更新序列、输入及CPU/CUDA RNG检查通过。每候选72次门检、两进程各651次完整层调用，共2748次；profile单独验证仍为一个kernel。该有限集不能证明所有输入/闭环质量；四权限保持false。

## HYPOTHESIS / UNKNOWN

大tile同时增加计算warp与shared，故其退化不能单独归因于warp数。固定tile对照排除了shared容量增大这一变化，却仍有小幅退化，说明“让等待warp参与MMA”并非充分优化。增加warp同时减小warp tile，改变shared读取/指令组织与寄存器活跃范围；哪一项贡献多少尚未测量。静态local bytes不等于实测local流量，occupancy两版均1CTA/SM，不能声称占用率提升。

本轮没有独立Cross K/V阶段计时或新NCU；表为完整层，不能把完整层退化比例写成投影阶段退化。上一轮NCU仍是4warp旧实现证据，不移植到8warp候选。

## RECOMMENDATION

保留较快的原4warp实现。若继续该方向，先对固定128tile的4/8warp Cross投影做相同前序下的阶段计时及MMA/shared/local计数，确定主循环、写回和编译影响；不要以“warp都在工作”作为筛选目标。小action投影未试8warp，不外推结论。跨阶段就绪填尾部仍排在原语原因之后。

## 资源与证据

本轮GPU进程生命周期79.587378秒，CPU构建/准备266.718863秒；M39累计2073.412951/36000秒。0作业异常/0超时，2性能负结果；worker均自然退出并回收，GPU0已释放，GPU1队列继续。原deadline未改。CUDA active未单独汇总，不能用进程生命周期冒充。远端证据约55.4MiB（报告写入前快照）。

- [原始计时汇总](TIMING-SUMMARY.csv)，两版各自TIMING-1/2.json与SIMPLE-GATE-RESULT.json。
- [选择/回退](SELECTION.json)、[资源](RESOURCE-AND-STOP-REPORT.json)，两版OWNED-CLEANUP.json。
- `candidate/`：大tile被测源码；`fixed-tile/candidate/`：固定tile被测源码；`before/`：原选中源码。
- `JOB.json`与`fixed-tile/JOB.json`分别登记，前者终态后才进行第二个隔离对照；未事后改原预算/终态。
- 活动路径：`算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh`；恢复4warp，baseline/原tile/投机生产与保护文件不动。
