# WAM短状态（2026-10-22T07:59:51.237516+08:00）

先读INDEX、CONTEXT-POLICY、memory约束，再按需读[Cycle四文件](cycles/wam-research-20261016/PROGRESS.md)；固定v15，不全量sync/全载旧索引。

## 主线与当前状态

M39先对齐完整单层，再根据原语/资源证据做tile ready调度。只改算子研究/wam_ops/optimized；baseline/原tile/投机生产和state/cycle/rules保护。当前没有30层融合或闭环性能放行。

[晨间结果](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)：三新候选各通过9真实签名/81调用有限逐位，但FA2强制内联、路由专化无稳定收益，全局compile+link LTO慢15.56%/16.15%/15.67%。全部撤回，源0cc1944b/DSOac3f9e46两机恢复夜间11。本机参考扩展旧ABI部署遗漏也已纠正，旧失败保留。

现用完整层B14短/中/长约2.296/2.453/2.476ms，相对夜间起点下降8.52%/9.75%/16.81%；同工作量uncached guarded2.070/2.187/2.201ms，仍慢10.94%/12.17%/12.50%。两进程Graph、有限真实输入；不是30层/完整chunk/机器人E2E，cached projection未含本轮。

现用11原9签名（B2action/B14action/B2video×3KV）+21结构变体有限逐位：Torch输出、23中间量、有效KV、普通/debug、输入/RNG。不能称全输入定理；08尾部旧FAILED与修复08b保留。本段现用数学/DSO未变，不重复写成新跑21变体。

## 新机制证据

当前版[NCU](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/local-ncu-reference-fixed/RESULT.md)：local读写243623424B（232.337MiB），active warp12.50%、Tensor47.93%、barrier warp stall40.67%；local不等于HBM，stall不是墙钟grid等待。无C/G PID/0MiB准入但NVML残留24–25%，仅诊断。

LTO中KV Cross-K/V嵌入574/570→772/768µs，独立约565/566µs；FFN1/2却199→179、138→125µs。插桩扰动0.23%–0.66%。说明编译上下文对不同阶段作用不同；确切spill/指令缓存贡献UNKNOWN。原统一资源干预也支持部分驻留限制，不能全部归因同步。

当前108CTA/256线程、100096B动态shared、255regs；仍无ready-DAG。[下一步](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/NEXT-ALIGNMENT-DECISION.md)：先限定FFN设备编译边界并保留Cross-K/V现有快路径，B14专用入口后备；都尚未实现。缓存命中时Cross-K/V不再执行，不能把不存在的tile当补位收益。

## 资源与其余研究

M39本段保守计费531.767秒、累计6004.497/36000秒，CPU准备952.151秒单列；本机/BAAI owned全退出。[终态](experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESOURCE-AND-STOP-REPORT.md)。原09:00截止保留；前瞻候选2→3修订、无收益与失败均保留。本段有限范围停止，不是预算耗尽，不自动追加新实验。

M40 RoboTwin工程4局完整：basket14双方成功−6.19%，microwave14双方失败−3.74%，受用摘要全程同；固定顺序单次探索性，非新质量N。累计54718.243980/86400秒、原10/22 01:26:04.706686已过，owned空。1800局/160GPU小时未启动/未批准；旧DEV90 T/C/P25/24/23各30，非独立TEST。

M17 LIBERO6000局（每方法四套×500）齐：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。暴露/首两S/长尾缺口保留，不是三seed重复或独立TEST。

四科学权限false、paused72/A_off/retention/旧FAILED不变，BAAI仅~/cxz，无新论文初稿。10/22 go/no-go，10/23冻结身份，10/28主表，11/01 18:00中文证据包；不足则收缩claim。
