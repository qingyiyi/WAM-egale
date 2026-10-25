---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-25T16:38:36.211404+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM 当前计划

投机重组小action负载→候选感知单层调度→质量约束下成本收益是待验证因果链。质量与系统并行；单层未集成机器人。四项科学权限均false。

| ID | 状态/下一动作 | 证据与依赖 |
|---|---|---|
| M42-MULTI-INIT | 本机固定Goal恢复准备中；完成后自动完整评估 | [resume05](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/PLAN.md)，本次T0 15:56:53、截止21:56:53；23剩余case/90臂，单worker、CPU2+2，每局1800秒/清理600秒 |
| M42-PAIRED-COST-RESULT | 等完整终态，汇总阈值质量/长度/接受深度，成本附带 | 旧10完整/7初态无效/1中断、0合格四臂组保留；不因恢复重跑旧INVALID或改门 |
| M40-EVAL100 | 双A100接续原600组三臂 | [原队列](../../experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)，160GPU小时/10天原预算；281/406旧缺口单列 |
| M40-MICROWAVE-FINAL-DIAG | 等微波炉100组完整后核官方Target及额外失败 | 不提前改变当前队列；不自动重试 |
| M39-SYNC-REVIEW | 继续FFN带读；同步合并点仅源码候选 | [审查](../../experiments/sm120-ffn-next-20261024/FFN-SYNC-SOURCE-REVIEW-20261025.md)，采用A不变；新GPU优化须有界登记/数值先行 |
| M17-EVIDENCE-GAP | 配对/FIT-DEV暴露缺口仍未解决 | 不借DEV补签独立TEST或正式性能许可 |
| W02 | 持续中文逐段写作，Prompt已交付 | [Prompt](../../prompts/WAM-PAPER-COLLABORATION.md)，不建本地论文正文 |
| H03 | 已完成短入口、主题路由与历史归档；不改实验 | [维护结果](../../maintenance/hw-compact-20261025/RESULT.md) |

本次闭合：M39-CLUSTER2-ATTRIBUTION、CLUSTER4-HYBRID、CTA-ATTRIBUTION、DSM-STREAM均保留负结果/独立A；M41五环境面板与临时资产清理、H02本机大文件归档已完成。结果见PROGRESS，不把单候选完成当整条研究完成。

10/22与10/23旧节点已过，不能补称按期达成；10/28主表、11/01 18:00北京时间中文证据包仍是规划节点。外部注册/摘要/正文日期使用前查官方，不据旧计划自动启动实验。

[精简前完整进度](PROGRESS.before-HW-compact-20261025.md)；[完整事件](EXECUTION.before-HW-compact-20261025.md)；[旧计划](PLAN.before-HW-compact-20261025.md)；[旧讨论](DISCUSSION-SUMMARY.before-HW-compact-20261025.md)。
