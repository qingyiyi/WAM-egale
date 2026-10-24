# 当前四MMA主循环有界诊断

状态：已收口，FINITE_OBSERVER_DIAGNOSTIC_PASS；自有GPU进程已退出。属于已有M39；只创建观察副本，当前采用代码和二进制保留。

- [x] CPU生成/审查；一次构建。
- [x] 三份真实完整block的数值、有效KV、输入和RNG核对。
- [x] 正常/OFF/COARSE/FINE完整层扰动，以及COARSE/FINE每tile循环包络扰动。
- [x] 稀疏观察TMA输入槽等待/发起、MMA输入ready等待、首次操作数加载、交错加载/MMA、输入槽释放。
- [x] 回传、清理、完整阶段表和HW结果。

原累计35746.41251785547/36000秒，余253.587秒。本轮GPU生命周期最多250秒（准备、编译、运行、回传、清理），单GPU单worker，CPU≤4，新增≤512MiB，完整block≤1200；一次构建，无算法改动、无自动重试。GPU在CPU材料就绪后才启用，T0登记绝对截止，预留45秒回传/清理；构建最多175秒，GPU worker最多30秒且受绝对截止保护。到限立即停止，不重置原账本。

固定当前四MMA／一TMA／三epilogue、64×128输出、48个K64、十段原BF16顺序、全部屏障。仅短/中/长KV288/1152/2160真实block0/B14/Q16，warm Cross K/V；不是episode或正式性能。

模式OFF不读钟；COARSE对每个被观察tile/warp只记录整个循环起止；FINE再稀疏记录K64内六个边界。每次观察tile编号模12的一组及step编号模12的一组，遍历后覆盖全部96tile/48K，重复2次。分别记录W0和W4–W7，不能只观察一个warp后称为全体。

完整层OFF相对正常、COARSE相对OFF、FINE相对COARSE任一绝对扰动>5%，或FINE/COARSE局部循环包络的均值变化>10%，结论标LIMITED_PERTURBATION，不把观察量当原路径精确占比。交错加载/转换/MMA无法用这些边界分成纯内存和纯Tensor Core时间，按合并区间报告；生产者与消费者并行，不相加当整层时间。保留全部失败，不通过不扩展候选。

当前采用版不改，不增加warp、不实现TMA输出store、不扩DAG。四质量/正式性能权限及production_admitted始终false。当前17阶段结构保持，旧比值明确历史，本轮只测上述诊断。
