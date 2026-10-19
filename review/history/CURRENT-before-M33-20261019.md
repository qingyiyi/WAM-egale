# 当前进度：一层完整Transformer block的tile/persistent

请固定本文件所在commit审阅；main仅本地核对后的进度镜像。历史4f04 paused72、b0dfac3科学基线与旧FAILED身份不变。review/RESULTS.json仍是历史paused72，不覆盖本页当前证据。

用户最新只要求**一层完整block**，在该层中研究tile/persistent。取消本轮30层迁移/验证，不作为完成门。远端~/cxz实现与验证，最终模型E2E后续本机。

M32已实现attention和FFN1阶段内部常驻CTA循环，完整self/cross/FFN、norm/RoPE/调制/残差与KV管理保留；QKV/Wo/FFN2仍cuBLASLt。没有跨阶段ready、warp专职分工或整层单kernel。baseline及依赖88项冻结，optimized独立编译。

九真实block输出/90连续有效KV状态/48原语逐tile检查通过；输入/RNG/参数/API通过。两进程探索计时：persistent+FFN1比同开FFN1静态路径B14慢4.42%–6.26%、B2 action慢2.56%–2.80%。没有新增加速，A/A最大10.07%保留。不是最强B14/Graph、完整chunk或E2E结果。五GPU作业77.927秒、731block、0full30/0episode，owned全部退出；计时guard PID命名空间误判和传输失败均保留。

先读：

1. [实现与结论](../.pipeline/experiments/persistent-tile-remote-20261019/RESULT.md)、[完整性能表](../.pipeline/experiments/persistent-tile-remote-20261019/PERFORMANCE.md)。
2. [源码说明](../算子研究/wam_ops/optimized/SUBGRAPHS.md)、[完整层](../算子研究/wam_ops/optimized/csrc/block.cu)、[常驻循环](../算子研究/wam_ops/optimized/csrc/persistent.cuh)。
3. [源身份与预算](../.pipeline/experiments/persistent-tile-remote-20261019/SOURCE-PERSISTENT.json)、[资源与止损](../.pipeline/experiments/persistent-tile-remote-20261019/RESOURCE-AND-STOP.md)、[当前HW](../.pipeline/PROGRESS.compact.md)。

下一步仅单层内部：先解释常驻开销，再选有真实复用价值的tile分组或相邻子图。不能由“用了persistent”推导更快或新贡献；不建议扩展30层来回避当前负结果。新GPU实验未启动。

四套既定三臂各500局计算已完成：Long T483/C484/P473；Goal463/464/467；Spatial472/466/464；Object497/498/499。Object取2026-10-19 21:06北京时间现场摘要，三个臂各500、同case齐；完整终态见[M32附带只读远端快照](../.pipeline/experiments/persistent-tile-remote-20261019/REMOTE-QUALITY-SNAPSHOT.json)。描述性覆盖不解除首两S配对FAILED、训练/DEV曝光、组件审计或长尾。四质量字段全部false，没启动独立TEST或正式E2E。

公开内容见[导出清单](OPERATOR-M32-EXPORT.json)。原PT/KV/权重/DSO/工具链未上传，GitHub不能独立重算这些张量。新源码本机旧DSO须重编，远端新DSO已验证。旧M31/M30/M28文档和原始结果仍保留，勿把旧完整模型门继承给新M32。
