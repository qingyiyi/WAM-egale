# 就绪tile调度：源码完成，GPU验证待新窗口

更新2026-10-22T12:16:15.511320+08:00。用户最新决定先做调度，替代此前继续逐阶段对齐的下一步顺序。仅修改optimized；baseline、原tile、投机生产不改。沿原M39，不创建新的研究容器。

## 已观察

- 新`ready_tiles=True`入口覆盖B14/Q16单层28个逻辑节点，原语tile大小和attention私有可见性保持原定义。普通调用仍用旧活动DSO。
- 原先每阶段的全grid集合改为完成计数与就绪领取。新kernel仅启动layout/队列初始化时一次`grid.sync()`；CTA/warp同步保留。
- hidden链优先，Cross-K/V在无可领关键路径tile时补位；Q/K/V与各自RMS/RoPE可在依赖完成后独立推进。
- FFN2第二段以每输出tile第一段完成为领取条件，保持原两段BF16舍入及bias顺序，不占CTA等待未就绪输入。
- CPU逻辑图24种配置通过：worker1/4/108，G1/G2任务数、有/无RoPE及Norm2；无任务遗漏/重复或逻辑死锁。该离散事件模型不是CUDA内存模型验证或速度预测。
- point、FFN1和attention内层数学体与抽取前逐字符比对保持；真正GPU逐位一致性仍待验证。
- CUDA12.4/SM80完整编译成功，耗时317.3秒；CPU加载成功且CUDA未初始化。candidate DSO `3bdf1a62f1c9807560d800de4bba0525afb3184a4020eaabd6a98ae89c26d4f1`。
- 远端首次CPU加载因默认解释器/runtime搜索路径缺libcudart失败；改用原va环境和既有CUDA12.4路径后CPU加载通过，未初始化CUDA。原失败记录保留。
- 链接后静态资源：新ready入口REG254、STACK456；本次编译顺序入口REG255、STACK936。这不是动态spill流量或速度结果，不能用entry阶段的32寄存器数字代替链接后的254。

## 实现边界与未知

普通节点暂等待整个输入节点完成，尚未实现全部逐行/逐head生产者消费者流水。领取单位是tile，执行单位是8warp CTA；各原语保留已有warp职责，未新增一个与GEMM长期并发的专职调度warp。

新路径仍是一个完整层kernel，activation保留原global workspace/BF16边界。仅针对uncached context；缓存命中时不存在的Cross-K/V不会被拿来虚增可调度任务，本轮未提供cached实现或收益。

FFN2不再并行计算同一输出tile的两个K段；第二段需第一段写回后才执行，可能减少并行度。另有原子队列、每tile发布fence、FA2描述符准备、设备调用和编译资源开销。收益、GPU内存可见性、输出/中间量/有效KV/RNG均为**待验证**，不借用旧11版本的通过结论。

数据先写后发布、消费者acquire和CTA共享空间复用同步依据[CUDA12.4.1内存同步说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#memory-fence-functions)实现；内存fence本身不能代替完整同步协议。

## 下一步

已准备9真实签名/81完整层调用、21结构变体/84调用及三KV两独立进程交错Graph对照。基线为旧11二进制及同工作量Torch/shared/guarded uncached路径。仅正确性通过才计时；不涉及机器人/30层/新质量证书。

原GPU窗口09:00已过，新45分钟窗口已向用户询问，尚未获答；没有启动GPU。仍计入原十小时，目前6004.497/36000秒，旧截止/账本不重置。原活动DSO保持`ac3f9e46…`。四科学权限保持false。

源码：[图](../../../../算子研究/wam_ops/optimized/csrc/ready_graph.h)、[执行器](../../../../算子研究/wam_ops/optimized/csrc/ready_tiles.cuh)、[原语/host](../../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)。

证据：[源manifest](SOURCE-MANIFEST.json)、[CPU图审查](CPU-AUDIT.json)、[编译终态](BUILD-STATUS.json)、[CPU加载](CPU-IMPORT.json)、[资源](RESOURCE-AND-STOP-REPORT.json)、[待执行GPU方案](JOB-PROPOSED.json)。原顺序源码在before保留，旧失败与速度证据不改。
