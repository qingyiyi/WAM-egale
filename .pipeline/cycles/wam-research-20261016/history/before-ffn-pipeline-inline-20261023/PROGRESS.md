---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T22:19:55+08:00'
current: M39
next: 当前FFN profile已完成，讨论FFN1尾部与额外local/指令；GPU1 RoboTwin初态补齐及自动队列继续跟进。
---

# 当前研究进度（2026-10-22T22:19:55.013997+08:00）

M39当前B源码不改，FFN profile完成。三KV两进程72整层/72独立时间线及本机NCU有限门通过。FFN1末组约207–209μs才就绪，前三组早约80μs；最后FFN2矩阵乘分段约92μs、写回18μs、前半段提交等待约1μs，退出收尾约6μs。主要问题不能解释成全局同步可直接去掉。动态local读写29.812MB、DRAM274.222MB对原Torch总220.216MB、整数指令约2.26倍；当前SASS中绝大多数LDL/STL在FFN流水函数入口/出口，支持ABI保存恢复来源；独立因果延迟份额UNKNOWN。NCU stall比例不是墙钟。

现有选定B无插桩独立FFN仍慢原Torch8.912%/9.084%/9.154%；完整层warm Graph原Torch/自有1.2185/1.7605/2.8473×，不是chunk/E2E。此次测量副本完整层扰动≤0.536%、资源相同，未替换入口。四科学权限及production_admitted仍false。GPU workers62.114秒、CPU编译147.252秒、保守265.018秒，累计12603.610/36000秒。owned退出，1个SSH启动通道超时但远端单次作业自然完成；无GPU失败/timeout。

M40新一期已获明确160GPU小时/最长10天授权。最新用户纠正分工：**GPU0固定留算子；GPU1独自接续1800局**，不自动借GPU0。T0=10/22 22:03:22+08:00，10天截止11/01 22:03:22；单worker160小时资源上限会更早触发。旧24小时账本54718.244秒/旧01:26截止不改。GPU1正在补每任务19→100有效初态；完成后CPU依赖核查、原GPU T5补全BF16 reset bank、T/C/P完整队列自动接续。每30局shard汇总，实施/初态/配对失败停止，不自动重试。当前尚无新的完整质量局。新面板为扩大DEV，旧FIT/CAL/DEV暴露标记保留。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | 当前FFN profile完成；生产B不改 | 和用户讨论FFN1尾部/额外local与指令开销，再做单因素优化 |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；非独立TEST，配对/暴露/长尾缺口保留 | 不恢复过期作业 |
| `M40` | RoboTwin扩大DEV | in_progress | 新160小时获准，GPU1实际准备有效初态，质量队列自动后继 | 跟进初态/bank/首完整三臂及分片汇总，GPU0不借用 |

[FFN结果](../../experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESULT.md)；[FFN资源](../../experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESOURCE-AND-STOP-REPORT.json)；[RoboTwin新计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/PLAN.md)。10/23身份冻结、10/28主表、11/01 18:00中文证据包；RoboTwin单卡估5–6天，预算上限不是保证完成时间。
