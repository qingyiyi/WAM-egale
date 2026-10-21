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

## 2026-10-21T11:44:03.604996+08:00 GPU1优先恢复 / M39诊断

- 用户明确优先RoboTwin，Maintain保存本轮接续授权；新CURRENT18已启动，真实首局成功，原24h累计预算不重置。
- 自动后继watch真实启动：完整18→审计→ON3000→CAL；DEV90未排队，不能宣称已完成迁移。
- M39同前序两进程完成，固定源未变；首失败11.329秒保留，总累计1649.673秒。Cross KV整层差距未因统一前序消失，动态spill根因仍未知。
- [RoboTwin](../../experiments/robotwin-pilot-20261020/current18-20261021/RESULT.md) / [算子诊断](../../experiments/single-layer-async-gemm-20261020/gemm-context-control-20261021-r2/RESULT.md)。
