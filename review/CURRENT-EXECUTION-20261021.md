# 当前执行证据（2026-10-21T17:05:55.995101+08:00）

**同计算体三组对照已经完成，672CTA没有比108CTA显著更快。** 固定256×128/8warp投影、同一个compiled kernel、三真实KV、两独立进程：普通672CTA/108静态/108原子同前序约615µs，原版约553µs；三调度中位数差≤0.17%，原子连清零仅增加0.17%–0.33%。独立投影仍慢11.09%–11.30%，不能把原版优势归于CTA数量/领任务。

36组投影逐位max_abs0、672tile各一次；B14整层23共有中间量+output、有效KV、输入/RNG有限核对通过。当前4warp源和DSO保持，大tile8warp完整层仍慢3.11%–5.00%。没有新NCU；独立local0/插桩整层local1456不能单独证明根因，插桩本身增加1.32%–1.40%。下一步GEMM主循环/写回对齐，再真实DAG。

- [完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md)、[逐进程计时](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/TIMING-SUMMARY.csv)、[诊断源码](../算子研究/wam_ops/optimized/csrc/projection_schedule.cu)。
- [资源/失败](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESOURCE-AND-STOP-REPORT.json)：本轮GPU61.040秒包含两计时脚本失败23.611秒；累计2134.453/36000秒。0timeout，owned已退出，GPU0释放。两失败与纠正源码保留，未伪造一次通过。
- 旧[8warp退化证据](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)和旧4warp NCU保留；不能把旧计数写作新8warp结果。

RoboTwin 17:02:19快照**45/90**完整，当前lift_pot-valid16-T，guard活跃；15组C/P初态/首两S审计通过。[巡检](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/MONITOR-20261021-170219.json)。GPU1自动接续，原24h累计及10/22 01:26:04硬截止不变，不自动重试。

四科学权限false；LIBERO/当前DEV不是独立TEST。baseline/原tile/投机生产/保护文件不改。原始张量/权重/DSO留服务器；旧paused72与review/RESULTS.json不修改。此前派生快照保留于commit `2180fdde7b103118bbffec3dc366c7a4e93d0159`，本入口只保留当前结论及证据指针。
