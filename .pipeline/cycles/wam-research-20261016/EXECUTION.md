# 最近执行检查点

## 2026-10-21T17:04:44.011067+08:00 同计算体三调度完成

- 同256×128/8warp compiled kernel，普通672、108静态和108原子同前序中位数差≤0.17%；含清零原子成本增加0.17%–0.33%。原版约553µs，自有约615µs，差11.09%–11.30%仍在计算体。
- 36投影组逐位与672tile各一次通过；两进程三B14完整层共有23中间量+output、有效KV/输入/RNG有限通过。原4warp源码/DSO保留；8warp完整层慢3.11%–5.00%。
- 两脚本错误（旧原版整层不可capture、raw K字段接口差）已修正并保留attempt1/2；成功两进程，无timeout。本轮GPU61.039747秒、CPU187.094542秒，累计2134.452699/36000秒。owned自然退出/reaped，GPU0释放。
- 插桩全层增加1.32%–1.40%，不将嵌入阶段差直接归local/sync。没有新NCU。
- M40 17:02:19完整45/90、15组C/P审计通过，GPU1继续，原截止不改。
- [结果与源](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md)、[账本](../../experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESOURCE-AND-STOP-REPORT.json)。

[前完整执行](history/projection-dispatch-before-20261021/EXECUTION.md)。
