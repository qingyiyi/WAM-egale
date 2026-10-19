# M32：一层完整Transformer block的tile/persistent实现与结果

状态：**实现完成，有限精度/状态门通过，新增persistent性能收益未成立**。用户最新范围是一层完整block；本轮完整30层调用为0，30层不是当前完成门。统计原件见[RESULT.json](RESULT.json)。

## 实际完成

在独立optimized中实现阶段内部的persistent CTA，self-attention、cross-attention和可选FFN1+GELU均调用显式tile设备函数。一个CTA依次执行 `id = CTA_id + n × grid_size`，每tile后排空cp.async并同步，再复用共享内存。默认网格为min(tile数,108个SM)。原private KV边界、在线softmax/合法KV split与combine、BF16舍入保持。完整层仍包含norm、RoPE、调制、投影、残差和KV管理。

它目前是同一层内分阶段的实现：QKV、输出投影、FFN2仍cuBLASLt；没有跨阶段ready队列、warp生产者/消费者分工或整层单kernel融合。固定步长分配本身不保证权重/KV复用。baseline及实际依赖88项冻结不变，投机源码/配置不改。

源码（本机根 `/rjs/cxz/论文探索`）：

- [完整层入口与三段子图](../../../算子研究/wam_ops/optimized/csrc/block.cu)：269/286/298行是self/cross/FFN，336行起依次调用。
- [persistent公共循环](../../../算子研究/wam_ops/optimized/csrc/persistent.cuh)。
- [attention任务执行](../../../算子研究/wam_ops/optimized/csrc/attention_tiles.cu)：57行 `persistent_attention`。
- [FFN1任务执行](../../../算子研究/wam_ops/optimized/csrc/ffn_tiles.cu)：60行 `persistent_linear`。
- [使用、DAG与边界](../../../算子研究/wam_ops/optimized/SUBGRAPHS.md)。

远端已编译工作区：`/root/cxz/ops/persistent-tile-20261019/workspace/算子研究`。只在~/cxz写入。新DSO在远端构建；本机旧M31 DSO没有自动重编，新Python入口会拒绝旧ABI。远端不加载带本机绝对依赖路径的冻结baseline DSO，以原Torch和独立optimized顺序路径作对照，不伪称同一baseline二进制已迁移验证。

## 有限正确性证据

九个真实完整block输入：action B2/B14×短中长，video B2 Q128×短中长。隐藏维3072，24头×128，FFN14336；action Q16，有效self-K288/1152/2160；本批video有效K272/1136/2144；cross-K512。每个B14 attention实例仍Q16，私有状态不跨候选共享。

[完整block原件](remote-evidence/jobs/block-state-1/results/RESULT.json)：九组原Torch输出与原保存gold逐位相同；optimized sequential/tiled/tiled_ffn/persistent/persistent_ffn的公开输出也逐位相同。tile/persistent的24个中间量与独立sequential比较；action融合路径23个（融合消去一个中间值）。这些中间量并非本轮独立逐项与Torch hooks重算。输入、CPU/CUDA RNG、参数地址/grad flags、安装还原通过。90项连续write1→write2→read0状态检查包含满池驱逐及全部逻辑有效KV；B14只读路径的有效状态另随九组输出检查。没有加载baseline模块或旧WAM动态库。

[逐tile原件](remote-evidence/jobs/tile-coverage-1/results/RESULT.json)：36项self/cross attention、12项FFN1，默认CTA数与强制2 CTA两种重放全部输出逐位；每tile恰好访问一次，每worker的实际完成数与任务分配相同。短B2/B14 profiler确认执行persistent_attention与persistent_linear；这些profile不作性能数据。

这些是真实单层输入的有限覆盖，不能证明任意输入、完整机器人接受路径、成功率或完整模型E2E。四项质量/TEST/非劣/正式性能字段保持false。

## 单层性能结果

[两独立进程全部九点表](PERFORMANCE.md)、[七入口CSV](BLOCK-TIMING.csv)。378个原始计时样本，另126次检查warmup。GPU0全过程单context，GPU1仍有Object；CPU4/5。窗口含动态KV管理、打包与同步；安装和cache恢复不计时。没有CUDA Graph，不与最强shared-prefix/O3/Graph混比。

persistent+FFN1相对同开FFN1的静态tile路径：B2 action慢2.56%–2.80%，B14 action慢4.42%–6.26%；Q128视频慢0.28%–1.08%。两进程方向一致，当前不推荐默认启用persistent作为性能优化。

原Torch A/A有一个10.07%异常点，保留；B14跨进程绝对耗时也波动。三轮中位只用于探索，不主张小幅稳定收益。原生顺序实现已有的工程收益不能归给新增persistent。

## 解释与下一步

OBSERVED：B14 self/cross各336个tile、FFN1为448个tile，persistent均限制为108个CTA；每CTA分配到多个独立tile。attention动态共享内存标准分支65536字节，split分支81920字节；FFN1为73728字节。编译资源见[KERNEL-RESOURCES.txt](remote-evidence/evidence/KERNEL-RESOURCES.txt)：FFN1 RoundedGelu静态154→persistent168寄存器/线程；部分attention栈/寄存器开销变化。编译值不等于实测occupancy，未测L2/DRAM与真实驻留。

HYPOTHESIS：缩小网格和循环/barrier可能损失并行度，且相邻任务还没有复用权重/公共历史；足以解释“不能因常驻就预期更快”的风险，但本轮没有完成因果profile，不能宣称唯一根因。

RECOMMENDATION：继续只研究这一层。先对实际attention/FFN1任务与资源做小范围profile，区分寄存器/驻留/尾部开销；再选择一种有复用价值的tile分组或生产者消费者子图。QKV/Wo/FFN2设备化及跨阶段就绪须各自满足原数值边界，不能为了凑整层单kernel降低精度门。静态路径保留为当前优化对照；不扩展30层，不无限网格搜索，不把现有循环命名成已证实MegaTransformer协同。

## 资源与失败

五个GPU进程生命周期共77.927秒、731次完整block、48次原语，0完整30层、0机器人episode，全部owned自然退出；其中一个计时资源守卫失败为12.781秒/0次block。原1800秒GPU/900次block/22:47停工作/22:52清理上限不重置。CUDA active时间未单独测量，记UNKNOWN。远端隔离目录约636.5MiB（逻辑字节，含工具链/源码/输入/构建；模型为已有目录的符号引用，不重复复制）。

网络准备有rsync缺失、慢tar中止、SFTP超时、反向隧道工作区900秒超时；原件全部保留。最终四并行压缩流补齐五个缺失的单层输入，九输入共用；未迁入real-verifier-fixture.pt。等待/传输计入19:52起墙钟，不混成GPU计算时间。详细[资源与止损](RESOURCE-AND-STOP.md)、[身份清理核对](FINAL-RESOURCE.json)、[源身份](SOURCE-PERSISTENT.json)、[冻结核对](FROZEN-AFTER.json)。
