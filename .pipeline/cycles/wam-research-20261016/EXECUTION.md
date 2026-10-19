---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-19T13:37:18.769422+08:00'
---
# 最近执行检查点

## 2026-10-19T13:37:18.769422+08:00：强基线与有限FFN原语收口

M26按本轮同意执行，独立算子目录；六真实block0，30eager/24Graph配置过，实际Linear M32→224、Q仍16。强cache trace FFN占21.9–29.9%。完整device FFN的FFN2六点失败；四个固定分块公式未复现，未改容差/阈值。FFN1+GELU收缩组件在两个独立进程六点严格过，完整block B2延迟降0.74–1.52%、B14 2.45–3.23%；保留Torch FFN2，无persistent/新30层或机器人作业。

17个受监督作业全部终态，751.141秒GPU生命周期（含setup/失败/NCU application重放），1093 block host/capture/replay intents、617原语；两工具失败原件保留，无timeout。GPU显存0/无可见compute PID，自有退出。理论CTA初始0是未SMEM opt-in的无效查询；真实输入执行后重读plain=2，fused achieved occupancy未知。

NCU命令漏显式clock-control none，本地help默认boost，实际改频UNKNOWN；原结果带配置偏差保留，后续bounded强制none/明确cache。NCU不进性能表，空卡利用率24%异常不改。baseline与1875投机源/配置复核不变，原模型字节与ca5f37e一致、block forward AST同原。所有四false不变。

[原始门/计时/失败/资源索引](../../experiments/operator-profile-device-20261019/RESULT.md)。Object13:31 T356/C156/P156，3worker活动、后继自动，无terminal；原预算不重置。下一步发布main和讲解；后续DAG/FFN2方案另收口。

更早完整执行记录见[本次历史](history/M26-before-result-20261019/EXECUTION.md)。
