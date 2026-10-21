# 当前执行证据（2026-10-21T18:01:48.432024+08:00）

**投影写回开销已部分解释并改善，收益进入完整单层。** 固定计算体的NCU与控制实验：shared accumulator stride128→136使写bank conflict从2,064,384变为0；四元素搬运进一步减少耗时。最终独立投影约577µs，同期Torch约553µs，仍慢约4.07%–4.26%。不能把本次实现修复包装成新调度机制。

当前4warp完整单层保持108CTA/阶段同步，只修改context投影写回。两独立进程B14短/中/长KV约2.660→2.560、2.872→2.774、3.124→3.025ms，耗时减少3.77%/3.42%/3.15%。9真实签名原Torch输出max_abs0、23共有中间量/有效KV/输入/RNG严格有限门通过。原始action生产不切换，尚未实现ready-DAG，不能称完整质量/E2E已通过。

- [完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md)、[机制与资料](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/DIAGNOSTIC-NOTES.md)、[逐进程完整层计时](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/whole-pad136/r2/SUMMARY.csv)。
- [精确选用源与DSO](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/SELECTION.json)，当前whole_layer源`fdf883e`；本地和远端复制同一远端实测DSO。
- [预算/失败](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESOURCE-AND-STOP-REPORT.json)：本轮GPU171.141秒，包含首次vec4无效差分对照30.471秒；另一次CPU编译失败0GPU。修复记录、旧失败保留。累计2305.594/36000秒，owned清理、GPU0释放。
- [代码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[epilogue](../算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh)。下一步剩余FFN/小投影及整层资源归因，再真实DAG。

RoboTwin 17:58:29快照完整**57/90**，当前`stack_blocks_two-valid17-C`，19组初态/首两S审计通过；[巡检](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-175829.json)。GPU1队列继续，24h累计及10/22 01:26:04.706686硬截止不变，无自动重试。

四科学权限false；LIBERO/当前DEV不是独立TEST。baseline/原tile/投机生产/保护文件不动。原始张量/权重/DSO留服务器；旧paused72与review/RESULTS.json不改。前派生快照在commit `32cf69b2d2a529839d465a5fc709d89f087b4987`。
