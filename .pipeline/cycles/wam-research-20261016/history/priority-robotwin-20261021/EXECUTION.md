# 当前执行检查点

## 2026-10-21T10:46:42.181704+08:00 M39

- 用户决定先小算子核对，澄清每阶段采用合适tile/任务数，再按DAG就绪填空；Maintain已合并到现有constraint。
- 官方源已绑定commit；12小算子三KV两进程独立逐位通过。point/RMS整层修改通过9签名有限门，固定预热两进程改善2.2%–3.0%。
- 全29阶段及CTA等待表已更新；空闲窗口/屏障尾部分列，不给纯同步百分比。原初轮漂移与所有负结果保留。
- 本轮GPU207.379秒；全部owned退出，SSH短时中断后已核验清理。RoboTwin未重启。
- [结果](../../experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md) / [前检查点](history/minor-ops-20261021/EXECUTION.md)。

## 2026-10-21T11:02:47.052904+08:00 三KV统计补全（仅CPU）

- 12类小算子×三KV×两独立进程，以及完整层三KV的全部进程范围已汇总。直接残差略慢、整层仍慢27%–47%均明确保留。
- Maintain补充算子汇报三档同时展示的要求；下一步先控制前序缓存条件，再定位Cross KV/FFN。没有新GPU作业。
- [短中长KV与两进程完整汇总](../../experiments/single-layer-async-gemm-20261020/minor-ops-20261021/THREE-KV-SUMMARY.md)。
