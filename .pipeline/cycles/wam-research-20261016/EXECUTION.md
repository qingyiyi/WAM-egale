---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-22T12:46:50.150956+08:00'
plan: PLAN.md
progress: PROGRESS.md
---
# 最近执行检查点

1. 明确45分钟授权登记，原累计十小时不增加。
2. 9签名81次、21结构84次有限精度/状态通过。
3. 三KV六臂两独立进程，ready慢55.72%–64.22%，性能NO-GO；同构建顺序入口近旧11。
4. clock插桩有限诊断通过，领取广播约21% CTA周期、发布约4%；不是GPU墙钟比例。两次入口错误完整保留。
5. 12:43:09核验全部owned退出；累计6895.808秒，当前不运行GPU。
6. 用户要求回退阶段同步并重新规划：本机源码恢复before，远端同步与身份核查；ready原始源码和DSO留证据，默认DSO从未替换。

[结果](../../experiments/single-layer-async-gemm-20261020/ready-tiles-20261022/RESULT.md)、[此前检查点](history/before-ready-gpu-result-20261022/EXECUTION.md)。
