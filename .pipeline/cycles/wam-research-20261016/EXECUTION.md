---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-25T18:38:14.572913+08:00'
---
# 最近执行检查点

| 时间（北京时间10/25） | 检查点与证据 |
|---|---|
| 18:35 | FFN1 1T/4M/3E W完成，三KV/两进程精度与计数通过，整层慢约0.8%，保留S；TMA store未实施。[结果](../../experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/REVIEW_RETURN.md) |
| 17:59 | FFN阶段拆分S完成并采用：三KV/两进程/27tensor/七段/状态/公开入口通过；2865调用，GPU清理。[结果](../../experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md) |
| 17:55–17:59 | Goal resume05闭合4合格四臂组；Robo569组/1707局，两卡继续。原失败/四false不改。 |
| 15:48 | C2/G2三KV/两进程与R0/R1完成，有限精度通过但NO_GO保留A；10413调用/17phase，105文件回传清理。[结果](../../experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md) |
| 15:45 | 本机旧窗口准备失败独立核对：VAE路径与self-kill，0teacher/新episode，204结果归档与临时资产清理；旧失败不覆盖。[核对](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume03/RECONCILE-RESULT.md) |
| 15:56–16:07 | 用户要求修复接续；新窗口至21:56，resume04计划性CPU传输交接后resume05沿同一T0/预算；原固定panel不改。[登记](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/REGISTRATION.json) |
| 16:14 | FFN同步CPU源审查，未修改采用A或产生新速度结果。[带读](../../experiments/sm120-ffn-next-20261024/FFN-SYNC-SOURCE-REVIEW-20261025.md) |
| 本次实查 | M42准备/双A100队列身份和进度，见[快照](../../maintenance/hw-compact-20261025/RESOURCE-SNAPSHOT.json)；快照不等于未来实时状态 |
| 本次H03 | 重建短入口、主题路由及Cycle摘要，旧原文/保护源/运行源逐项核验见[维护结果](../../maintenance/hw-compact-20261025/RESULT.md) |

完整原事件：[此前Execution](EXECUTION.before-HW-compact-20261025.md)，更早记录沿其归档链接读取。本文件只留检查点，不重复长汇报。
