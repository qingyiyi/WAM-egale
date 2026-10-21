# 投影写回优化：独立GEMM与整层结果

更新时间：2026-10-21T17:56:41.793869+08:00。本轮已完成并选择 **context-only pad136＋四元素搬运，保留原4warp整层**。两候选到此收口；没有进入异步DAG。baseline/原tile/投机生产未改，四科学权限false。

## 结论

NCU找到并通过控制修改验证了一项具体开销：原shared accumulator行stride128使K/V各出现2,064,384次shared写bank conflict；stride136后两次均0，shared写wavefront从2,752,512降为688,128。原版参考接近零。padding单因素减少独立投影3.49%–3.66%；四元素搬运在padding上再减少2.75%–2.93%。最终独立投影比同轮Torch慢4.07%–4.26%，没有完全对齐。

收益进入当前4warp单层后仍保留：两独立进程短/中/长KV分别减少3.771%–3.773%、3.352%–3.493%、3.116%–3.178%。9真实签名对原Torch输出max_abs0、23共有中间量逐位一致，全部有效KV、输入与RNG有限核对通过；不是对所有输入的保证，也不是机器人质量或E2E结果。

## 完整单层耗时

远端同GPU0、同输入、同进程交错before/after，各60预热、24样本，重复两个独立进程。表为两个进程各自中位数的均值，完整数据见whole-pad136/r2/TIMING-1/2.json及SUMMARY.csv。当前入口仍执行完整context K/V投影，未把跨chunk缓存收益当成新优化。计时与本机NCU分开，未引用profile延迟做性能表。

|历史KV|修改前整层ms|修改后整层ms|耗时减少|
|---|---:|---:|---:|
|288|2.660352|2.560000|3.772%|
|1152|2.872320|2.774016|3.422%|
|2160|3.123712|3.025408|3.147%|

这是修改前后自有完整单层对照，并非相对原Torch的完整层倍数；本轮原Torch负责精度与投影stage速度参考。当前NativeBlock debug入口不可直接Graph capture，不能将旧eager整层数混入本表。B2 action/video纳入正确性门，本轮整层性能仅B14；生产原始action依旧保留原路径。

## 独立投影控制实验

固定M7168/N3072/K3072，context=[14,512,3072]；短中长是历史288/1152/2160及真实输入差异，不是投影K改变。固定256×128×64 tile、8warp、672CTA，原三CTA调度负结果保留。两轮各两独立进程同前序对照；不同轮绝对数只并列，单因素结论由各轮同期对照得到，不把两轮乘积当完整层收益。

|KV|投影|修改前µs|paddingµs|padding＋四元素µs|本轮Torchµs|最终比Torch慢|
|---|---|---:|---:|---:|---:|---:|
|short|K|615.424|593.408|577.536|554.496|4.155%|
|short|V|615.424|593.920|576.512|552.960|4.259%|
|middle|K|615.424|593.920|576.512|553.984|4.067%|
|middle|V|615.936|593.920|576.512|553.728|4.115%|
|long|K|615.424|593.920|576.512|553.984|4.067%|
|long|V|615.424|593.408|576.512|553.472|4.163%|


pad136/RESULT-1/2和vec4/paired-r2/RESULT-1/2各24组有限投影审计，输出逐位、672tile各一次；hot/same-prefix分列，同前序采样时记录SM1410/memory1215MHz，hot存在降频，不用于主结论。独立三个版本资源均shared147472 bytes、254regs、localSize0、1CTA/SM。

## 为什么有效，尚未解决什么

- MMA主循环保持CUTLASS三级流水，未改变乘加或bias/BF16边界。改变的是shared accumulator排布及搬运宽度；padding错开不同lane quad的bank。
- SASS证实epilogue shared load由LDS.64改为LDS.128；全局输出仍是两条32-bit STG.E，不能说已经生成64-bit全局store。静态指令出现次数不等于动态计数，见SASS-STATIC-COUNTS.json。
- 整层仍108CTA、8物理warp，其中4个MMA warp；仍按原阶段同步。寄存器255/shared98848不变，localSize从1456变1464 bytes；尽管该字段略增，整层仍更快，不能单凭localSize判断性能或当作动态spill流量。
- 剩余独立投影约4%差距尚未归因；GEMM主循环调度、predicate及原版epilogue差异未分离。其他小投影/FFN与完整层资源开销仍需按同口径逐阶段核查。
- shared padding/向量化属于既有实现优化，不作为MegaTransformer新颖性；本轮没有新ready-DAG，不能声称已实现跨阶段异步。

## 数值/状态门

72次门检完整层调用，9签名：B2 action、B14 action、B2 video各短/中/长。原Torch直接forward、NativeBlock、候选输出逐位相同，23共有中间量逐位通过；重复运行稳定、输入/CPU-CUDA RNG未改变，全部有效KV状态一致。B2还核验update_cache=1→2→0合法状态序列。随后两进程B14完整层计时合计1026次调用，最终输出/共有中间量/有效KV/输入/RNG复核通过。仅已有代表fixture，未采新机器人数据。

## 选择与源码

当前设备代码：`算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh`；context调用在`whole_layer.cu`约171行，generic dispatch在`gemm_pipeline.cuh`。默认stride128维持其他stage；context专用stride136＋四元素路径。源码与精确选用DSO身份见SELECTION.json：whole_layer `fdf883e65d9dc87883f09588a7d7f9b72c0fb96c60f74f161f9efbd585952936`；cutlass_gemm `a390df2fde77fb28648a9222a799f6ee2bfaf091dd5583715d40516378772916`。本地和远端复制同一已验DSO，远端执行全门，本地本轮没有重跑整层门；投机生产不切换。

## 失败、资源与停止

保留两项实现/测量问题：vec4首次before/after误读相同二进制导致差分无效（30.471349 GPU秒）；首次整层遗漏generic dispatch模板导致CPU编译失败（17.848642秒、0GPU）。分别用明确不同路径的paired-r2、修正模板的whole r2完成；旧数据不删除、不改PASSED。另有一次启动前只读路径检查失败，无作业启动/GPU占用，已在PRELAUNCH-NOTE.md记录。

本轮GPU生命周期 **171.141460秒/1800秒**，CPU编译259.945245秒/2400秒；M39累计 **2305.594158/36000秒**，CUDA active时间UNKNOWN。0timeout，owned清理见OWNED-CLEANUP.json与local-ncu两STATUS；GPU0释放、RoboTwin GPU1继续。两个计算体候选用完，不追加搜索。原各job绝对截止/既有失败保留；本轮截止2026-10-21T19:25:53.417985+08:00。本地证据约3.44MiB，远端空间单独核验。

下一步：对当前选用版本核对剩余FFN/小投影原语与整层成本，解释剩余差距后，再实现有依赖就绪条件的tile提前执行。不能直接假定取消grid同步会解决余下问题。机制来源与详细profile边界见DIAGNOSTIC-NOTES.md。
