# WAM 投机推理与 verifier 优化

更新时间：2026-10-17T19:53:41+08:00。这是项目读取入口；实验结论以链接的原始结果与登记协议为准。

## 从这里恢复

1. 先读[上下文策略](CONTEXT-POLICY.md)和[记忆索引](memory/INDEX.md)；约束级规则必须读取，其他主题按需打开。
2. 当前研究：[进度](cycles/wam-research-20261016/PROGRESS.md)；本轮 HW 维护已完成：[归档进度](cycles/hw-maintenance-20261016/PROGRESS.md)。多个 active Cycle 时先按当前用户任务聚焦，不能静默选择最近一项。
3. 进入所选 Cycle 时读取其 PLAN、完整 PROGRESS、最近 EXECUTION 和 DISCUSSION-SUMMARY。避免扫描全部旧知识、报告和大 JSON。

## 两条研究路线

| 路线 | 当前结论 | 证据入口 |
|---|---|---|
| 投机采样 | M17本机Object单worker完整Target链启动中，Goal/Spatial等待AutoDL；M16 Long双卡四worker继续。90/fixture仍UNKNOWN，旧NO_GO/四false不变 | [当前状态](experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)、[评测范围](experiments/libero-fullscale-20261017/SCOPE-REVIEW.md) |
| tile / SDPA | 固定 PyTorch2.9 基线有限验收完成；没有新 MegaTransformer/verifier/E2E 收益 | [原最终报告](research/tile-sdpa/PYTORCH29_FINAL_20261012.md) |

算子原代码与 HW 保留；不恢复已撤回融合。网页规划已返回；当前主线、已结束作业及下一待登记诊断见[协同执行索引](research/speculative-sampling/WAM-COORDINATION-20261016.md)。

## 动态事实与历史

- 当前[PF2执行与12h批准](memory/global/requirements/decision-PF2源码执行与AutoDL十二小时窗口.md)已替代历史只读NONE/5GPUh帽；固定4f04 paused72与live77分列。M15十二局已清理；后续用户另已批准[全量与双卡至完成](memory/global/requirements/decision-AutoDL全量研究使用至完成.md)，不是借旧余额扩任务。AutoDL[无smoke/独立测试/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)，所有实验[默认优化reset](memory/global/requirements/decision-实验默认启用等价reset优化.md)。

- 历史旧链（非当前AutoDL）：[M12数值诊断](experiments/draft-teacher-path-20261016/RESULT.md)完成并清理，下一步只优先TF效果；MLP仅对照、算子后置。原50完整已清理、TF retention HOLD；[M11新诊断](experiments/task4-direction-20261016/EXPERIMENT.md)终态PARTIAL：1对完整风险复现，第二TF预算中断，自有worker/guard已清理。无120/waiter；AutoDL归还不再SSH，关机未知。
- [旧100状态](../投机采样/results/runtime/p0-autodl-old100-archive-watch-v1-20261015-r2/status.json)和[旧接续状态](../投机采样/results/runtime/p0-autodl-research-quality-dispatch-v1-20261015-r2/status.json)仅为历史证据，不能用作当前运行或自动启动依据。
- [Cycle 索引](cycles/INDEX.md)保存当前工作位置；[迁移与回退](maintenance/hw-upgrade-20261016/README.md)保存本轮维护证据。
- 原 PROGRESS、state/rules、知识228条及六份索引均保留，用于历史追溯。旧 M01–M04 completed 不代表整个研究完成。不要从旧历史“下一步”自动启动任务。

## 工具版本

账号默认已按用户明确授权切换到固定v15.0.0-alpha.2，旧安装保留；其他项目记录不自动迁移。`hw-codex --hw-legacy`提供旧技能兼容，完整回退见维护报告。新会话生效，自动Hook未信任时按本入口手动恢复，不依赖Hook。
