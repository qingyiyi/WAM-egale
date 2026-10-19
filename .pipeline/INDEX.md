# WAM 投机推理与 verifier 优化

更新时间：2026-10-19T18:37:38.352270+08:00。这是项目读取入口；实验结论以链接的原始结果与登记协议为准。

## 从这里恢复

1. 先读[上下文策略](CONTEXT-POLICY.md)和[记忆索引](memory/INDEX.md)；约束级规则必须读取，其他主题按需打开。
2. 当前研究：[进度](cycles/wam-research-20261016/PROGRESS.md)；本轮 HW 维护已完成：[归档进度](cycles/hw-maintenance-20261016/PROGRESS.md)。多个 active Cycle 时先按当前用户任务聚焦，不能静默选择最近一项。
3. 进入所选 Cycle 时读取其 PLAN、完整 PROGRESS、最近 EXECUTION 和 DISCUSSION-SUMMARY。避免扫描全部旧知识、报告和大 JSON。

[当前整体整理](cycles/wam-research-20261016/STATUS-20261019.md)：Long/Goal/Spatial首轮、B0有限GPU逐位、Object修复与原门缺口；旧paused72仅历史。

当前算子：[M32一层完整block的tile/persistent结果](experiments/persistent-tile-remote-20261019/RESULT.md)。9真实block/90状态/48原语有限逐位；新增persistent比静态更慢，尚无加速主张。当前只做一层，baseline及依赖冻结；[M30真实DAG](experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)作设计来源。最新质量完成数见当前Cycle，不以旧快照代替实时。

## 两条研究路线

| 路线 | 当前结论 | 证据入口 |
|---|---|---|
| 投机采样 | Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各500；Object14/准入/ON已完成、匹配Target200与全500双卡接续，本机只接算子；旧NO_GO/四false不变 | [资源迁移](experiments/libero-four-suite-20261017/RESOURCE-RELOCATION-20261018.md)、[当前Cycle](cycles/wam-research-20261016/PROGRESS.md) |
| 单层Transformer算子 | M32完整单层tile/persistent有限精度通过；新增persistent比静态更慢，当前只研究一层 | [M32结果](experiments/persistent-tile-remote-20261019/RESULT.md)、[源码入口](../算子研究/README.md) |

算子原代码与 HW 保留；用户最新允许独立修正版本并复用固定上游源码，见[M22](experiments/operator-torch-alignment-20261018/PLAN.md)；不恢复已撤回融合。网页规划已返回；当前主线、已结束作业及下一待登记诊断见[协同执行索引](research/speculative-sampling/WAM-COORDINATION-20261016.md)。

当前算子源码为独立[算子研究](../算子研究/README.md)，仅baseline/optimized；M28两份均完整native block，六actual30/24状态/原grad flags有限逐位过，两进程九点动态性能齐、owned退出。[结果与源码](experiments/block-native-baseline-20261019/RESULT.md)。最强B14/Graph、完整chunk/E2E尚未验证；M25/M26/M27旧门和失败保留。投机1875文件/原tile不改。[M29相关工作与创新边界](experiments/speculative-denoising-novelty-20261019/RESULT.md)确认多项直接先例，候选感知调度仍待实证。远端Object后继自动、同卡约束与最新完成数见[资源核对](experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md)，原M17预算不重置。

## 动态事实与历史

- [10/18晨报实际安排](experiments/long-morning-20261018/PLAN.md)：09:15统计/09:20回本机，未全500则PARTIAL；[成功率口径](experiments/libero-fullscale-20261017/SUCCESS-RATE-METHOD-20261018.md)。[最终500组汇总](experiments/long-morning-20261018/FINAL-SUMMARY.md)已交付；三轮均值为后续偏好，后两轮尚未登记。


- 当前[PF2执行与12h批准](memory/global/requirements/decision-PF2源码执行与AutoDL十二小时窗口.md)已替代历史只读NONE/5GPUh帽；固定4f04 paused72与live77分列。M15十二局已清理；后续用户另已批准[全量与双卡至完成](memory/global/requirements/decision-AutoDL全量研究使用至完成.md)，不是借旧余额扩任务。AutoDL[无smoke/独立测试/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)，所有实验[默认优化reset](memory/global/requirements/decision-实验默认启用等价reset优化.md)。

- 历史旧链（非当前AutoDL）：[M12数值诊断](experiments/draft-teacher-path-20261016/RESULT.md)完成并清理，下一步只优先TF效果；MLP仅对照、算子后置。原50完整已清理、TF retention HOLD；[M11新诊断](experiments/task4-direction-20261016/EXPERIMENT.md)终态PARTIAL：1对完整风险复现，第二TF预算中断，自有worker/guard已清理。无120/waiter；AutoDL归还不再SSH，关机未知。
- [旧100状态](../投机采样/results/runtime/p0-autodl-old100-archive-watch-v1-20261015-r2/status.json)和[旧接续状态](../投机采样/results/runtime/p0-autodl-research-quality-dispatch-v1-20261015-r2/status.json)仅为历史证据，不能用作当前运行或自动启动依据。
- [Cycle 索引](cycles/INDEX.md)保存当前工作位置；[迁移与回退](maintenance/hw-upgrade-20261016/README.md)保存本轮维护证据。
- 原 PROGRESS、state/rules、知识228条及六份索引均保留，用于历史追溯。旧 M01–M04 completed 不代表整个研究完成。不要从旧历史“下一步”自动启动任务。

## GitHub发布

仅维护[main当前进度镜像](https://github.com/qingyiyi/WAM-egale/tree/main)，直接发布本地核对后的代码/HW；旧固定commit仍保留。[收拢结果](maintenance/github-main-20261018/RESULT.md)。

## 工具版本

账号默认已按用户明确授权切换到固定v15.0.0-alpha.2，旧安装保留；其他项目记录不自动迁移。`hw-codex --hw-legacy`提供旧技能兼容，完整回退见维护报告。新会话生效，自动Hook未信任时按本入口手动恢复，不依赖Hook。

历史算子核查：[M27](experiments/operator-shortkv-b14-20261019/RESULT.md)区分默认复现/Torch与FFN融合收益；后者B14约3%是耗时下降。源码完整路径/实现与测量范围见[代码导航](../算子研究/README.md)。新增作业PARTIAL/等待到期已清理，不改原门。
