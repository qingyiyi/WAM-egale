# WAM 投机推理与 verifier 优化

更新时间：2026-10-16T15:21:49.997299+08:00。这是项目读取入口；实验结论以链接的原始结果与登记协议为准。

## 从这里恢复

1. 先读[上下文策略](CONTEXT-POLICY.md)和[记忆索引](memory/INDEX.md)；约束级规则必须读取，其他主题按需打开。
2. 当前研究：[进度](cycles/wam-research-20261016/PROGRESS.md)；本轮 HW 维护已完成：[归档进度](cycles/hw-maintenance-20261016/PROGRESS.md)。多个 active Cycle 时先按当前用户任务聚焦，不能静默选择最近一项。
3. 进入所选 Cycle 时读取其 PLAN、完整 PROGRESS、最近 EXECUTION 和 DISCUSSION-SUMMARY。避免扫描全部旧知识、报告和大 JSON。

## 两条研究路线

| 路线 | 当前结论 | 证据入口 |
|---|---|---|
| 投机采样 | TF64结构/B14接口暂冻供算子；CONT3000权重/配方未过风险保留门，扩大120未注册；A_off不变、四权限false | [当前研究恢复](cycles/wam-research-20261016/PROGRESS.md)、[今天50完整结果](research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md) |
| tile / SDPA | 固定 PyTorch2.9 基线有限验收完成；没有新 MegaTransformer/verifier/E2E 收益 | [原最终报告](research/tile-sdpa/PYTORCH29_FINAL_20261012.md) |

算子原代码与 HW 保留；不恢复已撤回融合。网页规划已返回；当前主线、已结束作业及下一待登记诊断见[协同执行索引](research/speculative-sampling/WAM-COORDINATION-20261016.md)。

## 动态事实与历史

- 当前以[本轮终态/下一步](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/status.json)为准：50完整已清理、TF retention HOLD，无GPU作业/CPU waiter；AutoDL已归还不再SSH，实例关闭未知。
- [旧100状态](../投机采样/results/runtime/p0-autodl-old100-archive-watch-v1-20261015-r2/status.json)和[旧接续状态](../投机采样/results/runtime/p0-autodl-research-quality-dispatch-v1-20261015-r2/status.json)仅为历史证据，不能用作当前运行或自动启动依据。
- [Cycle 索引](cycles/INDEX.md)保存当前工作位置；[迁移与回退](../review/OMISSIONS.md)保存本轮维护证据。
- 原 PROGRESS、state/rules、知识228条及六份索引均保留，用于历史追溯。旧 M01–M04 completed 不代表整个研究完成。不要从旧历史“下一步”自动启动任务。

## 工具版本

账号默认已按用户明确授权切换到固定v15.0.0-alpha.2，旧安装保留；其他项目记录不自动迁移。`hw-codex --hw-legacy`提供旧技能兼容，完整回退见维护报告。新会话生效，自动Hook未信任时按本入口手动恢复，不依赖Hook。
