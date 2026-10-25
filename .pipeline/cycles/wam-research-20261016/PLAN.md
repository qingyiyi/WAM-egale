---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-25T19:36:51.428062+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM 当前计划

投机重组小action负载→候选感知单层调度→质量约束下成本收益是待验证因果链。质量与系统并行；单层未集成机器人。四项科学权限均false。

| ID | 状态/下一动作 | 证据与依赖 |
|---|---|---|
| M42-MULTI-INIT | resume05已闭合，81attempt、4合格完整四臂组 | [本轮结果](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/RESULT.md)，原T0/21:56截止与全部INVALID保留，进程已清理 |
| M42-PAIRED-COST-RESULT | 待核4组的任务覆盖、门控/长度分层；不可直接定最终阈值 | 54完整均成功，27其他attempt及完整未配齐样本单列；仍是不平衡暴露DEV |
| M40-EVAL100 | 双A100接续原600组三臂 | [原队列](../../experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)，160GPU小时/10天原预算；281/406旧缺口单列 |
| M40-MICROWAVE-FINAL-DIAG | 等微波炉100组完整后核官方Target及额外失败 | 不提前改变当前队列；不自动重试 |
| M39-FFN1-CLUSTER2-INPUT | 已完成：共享A比B快19%–21%，仍比S慢22%–26%，保留S | [结果](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/REVIEW_RETURN.md)，三KV/两进程/5400调用；等待诊断与清理完成 |
| M39-FFN1-CLUSTER2-WEIGHT | 已完成；D精度通过但整层慢49.89%–60.62%，保留S | [结果](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/REVIEW_RETURN.md)，等待定位/三KV/两进程/4836调用，清理完成 |
| M39-FFN1-WARP-PIPELINE | 已完成，精度通过但整层慢0.79%–0.85%，保留S | [结果](../../experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/REVIEW_RETURN.md)，三KV/两进程；mbarrier热循环，TMA输出尚未实现 |
| M39-FFN-STAGE-SPLIT | 已完成并采用S；后续W局部流水负结果保留，先定位同tile输入/结果等待与写回 | [结果](../../experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)，2865调用、三KV/两进程/公开入口精度通过；完整层相对A延迟下降约1.8%–2.3%；17:59清理，不自动开下一候选 |
| M17-EVIDENCE-GAP | 配对/FIT-DEV暴露缺口仍未解决 | 不借DEV补签独立TEST或正式性能许可 |
| W02 | 持续中文逐段写作，Prompt已交付 | [Prompt](../../prompts/WAM-PAPER-COLLABORATION.md)，不建本地论文正文 |
| H03 | 已完成短入口、主题路由与历史归档；不改实验 | [维护结果](../../maintenance/hw-compact-20261025/RESULT.md) |

本次闭合：M39-CLUSTER2-ATTRIBUTION、CLUSTER4-HYBRID、CTA-ATTRIBUTION、DSM-STREAM均保留负结果/独立A；M41五环境面板与临时资产清理、H02本机大文件归档已完成。结果见PROGRESS，不把单候选完成当整条研究完成。

10/22与10/23旧节点已过，不能补称按期达成；10/28主表、11/01 18:00北京时间中文证据包仍是规划节点。外部注册/摘要/正文日期使用前查官方，不据旧计划自动启动实验。

[精简前完整进度](PROGRESS.before-HW-compact-20261025.md)；[完整事件](EXECUTION.before-HW-compact-20261025.md)；[旧计划](PLAN.before-HW-compact-20261025.md)；[旧讨论](DISCUSSION-SUMMARY.before-HW-compact-20261025.md)。
