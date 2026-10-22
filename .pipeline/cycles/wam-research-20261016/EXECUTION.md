---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-22T14:42:31.869291+08:00'
plan: PLAN.md
progress: PROGRESS.md
---
# 最近执行检查点

1. HW Maintain保存用户按需调制融合及重测决定，仅本轮单因素，不恢复ready。
2. 修改optimized单CUDA源：modulation4保留FP32加法，普通MOD不物化，debug只用于审计。
3. CPU编译303.392秒；上传失败/超时保留，未改GPU截止。
4. 81+84完整层有限逐位/状态通过；两进程各9408调用，三KV约1%同向改善。
5. 14:38:38核验owned全部退出；GPU worker94.626秒、保守整窗430.874秒，累计7326.682秒。
6. 两机源9edf85a2/DSO41af9a65一致，采用新版；阶段表已更新，其他阶段尚未融合。

[结果](../../experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)、[前检查点](history/before-modulation-result-20261022/EXECUTION.md)。
