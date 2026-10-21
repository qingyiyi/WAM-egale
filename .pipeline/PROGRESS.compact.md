# WAM短状态（2026-10-22T02:46:38.498460+08:00）

先读INDEX、CONTEXT-POLICY、memory约束，再按需读[Cycle四文件](cycles/wam-research-20261016/PROGRESS.md)；v15语义入口，不全量sync或全载旧索引。

## 当前主线

M39先对齐单层，再依据原语/资源证据做tile ready调度。只改算子研究/wam_ops/optimized，baseline/原tile/投机生产不动。当前不是30层大融合，不是闭环性能放行。

## 算子：已完成和未完成

[夜间结果](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)：12实现候选＋08b修复已终态，最终选择11。B14短/中/长KV完整层约2.296/2.453/2.476ms，比夜间开始下降8.52%/9.75%/16.81%；同轮uncached guarded强对照2.070/2.187/2.201ms，仍慢10.94%/12.17%/12.50%，未完全对齐。2独立进程/轮换Graph，非profiler计时、非机器人E2E。

9真实签名与21结构变体通过有限逐位：原Torch输出、23共有中间量、有效KV、普通/debug、输入/RNG。08 short/drop1的2个中间元素差1.49e-8旧FAILED保留，08b非16尾G1修复。最终源0cc1944b/DSOac3f9e46两机同一验证二进制，旧2a8保留。仍108CTA/256线程单层；静态分派/padded写回/双4warp attention/FFN1 N256K32/移除冗余入口同步，真实出口依赖保留，ready-DAG未实现。

[逐阶段](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/stage-control-final-r2/RESULT.md)：self-attention、Cross-K RMS、FFN/GELU为剩余显著项；独立阶段仍慢，不能全怪grid同步。资源控制支持统一shared/驻留限制是部分原因。静态local936B不是动态spill。本机最终版NCU新窗口等待自然空闲，最迟08:47:30启动、09:00清理；前窗口外部干扰0kernel，原PARTIAL保留。

M39累计保守计费4850.877/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。 09:00硬截止，本轮30000秒/新增3GiB不增。[资源状态](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.json)。远端GPU作业已退出；本机仅等待/诊断，不清理别人进程。

## RoboTwin与LIBERO

M40[工程跨任务4局](experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)已终态，basket14双方成功−6.19%、microwave14双方失败−3.74%，全程受用观察/物理/RNG/命令/RPC摘要同；固定顺序单次探索性，非质量N。旧组合valid16−7.33%保留。累计54718.243980/86400秒、原10/22 01:26:04.706686截止已过、owned空；1800局/160GPU小时未启动/未批准。本机最多1/4并回传远端，当前0worker。旧DEV90 T/C/P25/24/23各30局，非独立TEST；提交与真实推进分列。

M17 LIBERO6000局（每方法四套×500）齐；Long T/C/P483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。曝光/首两S/长尾缺口仍在，不是每方法三seed重复，不是独立TEST。

## 不变边界与日期

四项科学权限false；paused72、A_off、旧retention/FAILED、baseline/原tile/投机生产和state/cycle/rules不动。BAAI仅~/cxz；未新建论文初稿。 TF64结构仅研究暂冻，旧TF质量NO-GO不因算子逐位通过而改变；PF2不是bonus。BAAI仅~/cxz。10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包；证据不足缩小claim。
