# 当前执行：ready验证后回退，重新规划

新ready路径有限精度/状态通过，但完整层短/中/长3.765/3.887/3.858ms，旧11为2.292/2.457/2.477ms，慢64.22%/58.20%/55.72%。同构建顺序入口近旧11，不能把退化归于简单源码抽取。clock64插桩扰动0.46%–1.21%，领取广播约21%、发布约4%的CTA累计周期；不是墙钟占比。

用户要求恢复逐阶段全局同步并重新顶层规划；两机源码已恢复0cc1944b、活动DSO始终ac3f9e46。ready源码/DSO与失败留证据，不在当前活动路径。当前28数学阶段、39次grid同步；约1.4万逻辑tile多数静态分配，不等于1.4万次队列领取。

[完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)、[阶段公式/tile/同步](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[当前源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[HW](../.pipeline/PROGRESS.compact.md)。新45分钟窗口已于12:43清理，累计6895.808/36000秒，不自动开新GPU任务。四科学权限false、旧paused72身份不改。
