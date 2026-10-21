# 当前执行证据（2026-10-21T20:11:05.142202+08:00）

**当前源剩余GEMM核对完成，活动数学实现未再改。** Cross K/V与FFN独立/整层设备调用点差约±1.5%；小投影仍有约11%–58%的额外集成成本。FFN1自身比Torch慢约34%–35%，FFN2约25%–27%。下一步先固定形状的小投影设备函数，随后FFN原语，最后DAG；本轮未实现新候选。

- [完整表与边界](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESULT.md)、[逐格CSV](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/STEADY-STAGE-COMPARISON.csv)。三KV/两进程，原短KV升频样本保留，固定600预热补短KV；插桩扰动约−0.30%至+0.06%。
- [为什么写bank conflict为0](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/BANK-CONFLICT-EXPLANATION.md)：stride128→136将四行bank错开；源码枚举预计额外2064384→0，与旧独立K/V NCU一致。不是整层所有冲突为0，也不是性能误差为0。
- [有限精度](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/PRECISION-SUMMARY.json)：320阶段输出比较max_abs0、8组有效状态检查；前轮九签名门仍保留，没有新增30层/闭环质量证书。
- [资源与停止](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESOURCE-AND-STOP-REPORT.json)：GPU68.215秒，累计2373.809/36000秒。0GPU失败/timeout；远端cuobjdump缺失的CPU失败改用同DSO本地反汇编，原记录保留。owned退出、GPU0释放。
- [当前主实现](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[写回实现](../算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh)。保留前轮context-only padding与vec4，当前源fdf883e，baseline/原tile/投机生产不动。

RoboTwin 20:07:50完整**89/90**，`open_microwave-valid18-C`，29组C/P审计通过；[巡检](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-200750.json)。GPU1原队列继续，24h累计与10/22 01:26:04.706686截止不变，不自动重试。

四科学权限false；LIBERO与DEV不是独立TEST。旧paused72与review/RESULTS.json保留。原始输入/模型/DSO和完整SASS留服务器；公开派生表不能声称离线重验未公开张量。前派生快照commit `ca5354a35310123ce258af3d987b63572bf83181`。
