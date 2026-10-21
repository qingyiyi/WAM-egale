# 当前执行检查点

## 2026-10-21T12:25:51.468246+08:00 M39 两项完成

- 用户批准先Cross KV设备函数控制、再FFN；仅optimized数学源修改，tile/舍入/sync固定。
- 两候选9签名有限门通过；最终直接两进程三KV完整层下降9.46%–10.87%，仍慢15%–31%。一个kernel，没有ready。
- SASS与阶段profile支持函数组织是Cross开销重要来源；插桩约±1%，动态counter未知。FFN独立增量约0.6%，不夸大。
- 本轮GPU120.995秒、CPU构建523.226秒；累计1770.668/36000秒；所有owned退出。GPU1 CURRENT18/watch继续。
- [本轮报告](../../experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md)；[前执行检查点](history/gemm-function-control-20261021/EXECUTION.md)。
