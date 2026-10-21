# 同计算体CTA调度对照结果

更新时间：2026-10-21T17:01:59.151538+08:00。M39有界诊断已完成；两次脚本失败保留。当前4warp整层未替换，四权限false。

## 结论

**在本次真实B14 Cross K/V输入上，普通672CTA并没有显著快于108常驻CTA。** 三种调度使用同一个compiled kernel、同256×128×64 tile、8个MMA warp、相同CUTLASS三级主循环、同遍历编号与BF16 bias/写回边界。不能再把原版的优势解释为“672比108多”。

同原版前序条件下，两独立进程各三KV×K/V共12个对照，中位数：静态108相对普通672变化−0.1664%～+0.0831%；原子108相对普通变化−0.0831%～0%。队列清零和相同marker边界一起计入后，原子108增加0.1660%～0.3322%。这只支持未观察到有意义的调度差异，不是所有shape都等价的证明。

自有独立投影仍比原版stage慢11.09%～11.30%。当前主要待查的是投影计算体、访存流水和epilogue，不能用改CTA数量解释或弥补该差距。寄存器/CTA驻留变化不是独立已证根因。

## 同前序投影耗时

单位µs。表中为两个独立进程各12次样本中位数的中位数；全体样本/逐进程范围见RESULT-1/2.json和TIMING-SUMMARY.csv。原版参考为相同真实输入的原生stage入口，无额外跨候选注意力。

|历史KV|投影|原版stage|普通672CTA|108静态|108原子|插桩整层8warp内该阶段|
|---|---|---:|---:|---:|---:|---:|
|short|Cross K|553.984|615.680|615.680|615.424|633.344|
|short|Cross V|552.960|615.424|615.424|615.424|630.528|
|middle|Cross K|553.984|615.424|615.424|615.424|633.600|
|middle|Cross V|552.960|615.424|615.424|615.424|631.040|
|long|Cross K|553.728|615.424|614.912|615.424|633.600|
|long|Cross V|553.472|615.424|614.912|615.424|631.040|

历史KV分别288/1152/2160；投影的GEMM始终M7168、N3072、K3072，context=[14,512,3072]。历史变长不等于GEMM的K变长；本矩阵用于核对不同真实输入及其前序访存。普通672是一个launch中的总CTA数；三调度均每CTA256线程，实际资源只允许1CTA/SM。原子队列会包含退出领取，不把counter末值当合法tile数。

计时区间为CUDA Graph中同样的前后globaltimer marker，投影输出预分配；native阶段分配由Graph池预置。主kernel区间不含队列zero，另列reset与inclusive区间；不减空marker。hot与same-prefix分列，不能混合缓存状态。主同前序与整层全部采到SM1410MHz/memory1215MHz；hot有1380/1395MHz，作为次级诊断保留，不用其细微差异宣称优化。GPU0每长度前后独占核查，GPU1 RoboTwin并行、同机CPU2线程。非机器人正式性能。

## 独立与嵌入的边界

独立同计算体：shared147472 bytes，254 regs，localSize0，1CTA/SM；插桩8warp整层：shared148000、255 regs、localSize1456，1CTA/SM。独立编译无stack/spill；这不能推导整层local全部是溢出或全部来自该投影。

插桩整层的Cross K约633µs、V约631µs，相对独立约615µs还多15～19µs。但整层插桩本身增加1.315%～1.402%（约36～46µs/完整层），而且embedded包含阶段队列、额外CTA/grid同步及设备函数上下文。因此这些阶段差只是描述性诊断，**不能将其全部归于local memory，也不能独立断言真实嵌入惩罚就是2%～3%**。若继续分摊，需要更低扰动的原语与SASS/NCU证据。

同轮完整层Graph复核（非原版整层E2E）：

|进程|KV|当前4warp ms|已拒绝大tile8warp ms|8warp变慢|插桩额外开销|
|---|---|---:|---:|---:|---:|
|1|short|2.670192|2.754496|3.157%|1.341%|
|1|middle|2.883088|3.000288|4.065%|1.315%|
|1|long|3.132352|3.289088|5.004%|1.354%|
|2|short|2.671024|2.754048|3.108%|1.320%|
|2|middle|2.886672|3.001184|3.967%|1.319%|
|2|long|3.132640|3.287040|4.929%|1.402%|

原版整层NativeBlock不可直接在当前入口capture，本轮不提供其整层Graph速度，不拿旧eager数字混算新加速比；原版整层继续作为精度参考。两次失败详见HARNESS-CORRECTIONS.md。旧8warp性能负结果未覆盖。

## 精度与覆盖

- 三KV×K/V×三调度×两进程=36组真实投影：逐位通过、max_abs0；每组672个tile各执行一次。审计时NaN初始化输出，visits/owner仅审计，不在计时热路径开启。
- hot/同前序两条件、四臂的输出及前序缓冲有限逐位核对全部通过。raw K/V由原版stage独立计算参考；不伪称原版整层debug保存了不存在的字段。
- 两进程各三真实B14完整层；active4/large8/trace8的23个共有中间量+output逐位通过，全部有效KV、输入与CPU/CUDA RNG不变。成功attempt合计1326次完整层调用、61368次stage调用（含前序与预热），失败调用另保留。
- 仅本次B14单层fixtures，不是所有输入证明、完整30层或机器人闭环质量。没有修改默认A_off、retention、tau或原FAILED。

## 源码与证据

- 诊断入口：`算子研究/wam_ops/optimized/csrc/projection_schedule.cu`；mode0/1/2共用同一个projection内核。
- 固定设备计算体：`optimized/csrc/projection_tile.cuh`，复制自前轮已核对的256tile/8warp候选；构建：`optimized/build_projection_schedule.py`。
- 当前整层仍为`optimized/csrc/whole_layer.cu`，SHA256 `df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a`。远端active DSO逐字节等于选用4warp归档。baseline/原tile/投机生产未改。
- JOB/SOURCE-MANIFEST、worker、RESULT-1/2、SUMMARY、FINITE-GATE、TIMING/FULL/EMBEDDED-SUMMARY及OWNED-CLEANUP共同构成证据；attempt1/2保留原错误与源码。原始fixture/权重/DSO在服务器，公开镜像仅小证据。

## 资源及下一步

本轮GPU生命周期61.039747秒（含两失败23.611243秒），CPU编译187.094542秒；M39累计2134.452699/36000秒。无timeout，owned进程自然退出/reaped，GPU0释放。CUDA active时间UNKNOWN，不用生命周期替代。新证据约26.73MiB，小于2GiB上限；18:14:07旧登记截止未改。

下一步先固定该可复现投影计算体，将CUTLASS主循环/epilogue与Torch采用的cuBLAS路径按同shape、同精度边界对齐。对当前8warp版本补必要的NCU/SASS证据后再挑一个修改，不能移用旧4warp的计数作本轮根因。CTA领取已经有直接负结果，暂不继续扩大CTA/warp搜索。原语差距收敛或可解释后，再实现独立ready tile填补尾波；本轮未实现异步DAG，也不因此取消用户的单kernel目标。
