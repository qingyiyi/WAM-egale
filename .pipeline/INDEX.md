# WAM 投机与单层 Transformer 研究入口

更新：2026-10-20T15:01:42+08:00。

先读[上下文策略](CONTEXT-POLICY.md)、[记忆索引](memory/INDEX.md)及约束级记忆。当前任务聚焦[wam-research Cycle](cycles/wam-research-20261016/PROGRESS.md)，读取其PLAN、完整PROGRESS、最近EXECUTION和DISCUSSION-SUMMARY；[短状态](PROGRESS.compact.md)。不默认展开旧知识大索引/trace/大JSON。HW维护Cycle已归档，不代表当前科研完成。

## 当前结论

- [单层最终结果](experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：九真实签名×两路径有限逐位通过，每路径18合法KV转移对原Torch通过；当前默认仍慢native2.68–5.02倍，ready额外慢1.2%–6.0%。[源码](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)/[原语](experiments/single-layer-async-gemm-20261020/PRIMITIVE-EFFICIENCY-RESULT.md)/[ready负结果](experiments/single-layer-async-gemm-20261020/ROW-READY-RESULT.md)。
- [本轮资源/止损](experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)：1380/1500block，43owned根进程回收，14:12现场两卡无compute进程。旧M39资源FAILED和早期数值/性能负结果保持，有限新PASS不追溯改旧记录。
- [RoboTwin原渲染已恢复](experiments/robotwin-pilot-20261020/DRIVER-INTEROP-REPAIR-RESULT.md)：A100/535配套库两次完整专家/四路观察通过；595 compat路径复现非法访存。仅客户端环境修复，系统驱动不变，旧失败保留，0policy/teacher。下一步首6条Target，无自动GPU队列。
- [MegaMoE差距审查](experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：已有单kernel/warp分工，仍缺K片段流水和中间槽生命周期；B14专用化存在32KiB无用宽路径缓冲机会，性能收益未测。
- [四套原质量](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：Long T483/C484/P473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各500；首两S FAILED/训练DEV曝光/长尾等原门不变。[原成本](experiments/libero-four-suite-performance-20261019/RESULT.md)/[新同卡330/360](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)。不是独立TEST/非劣。

[两卡执行决定](memory/global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)已替代旧仅规划，详细依赖/预算见[唯一计划](cycles/wam-research-20261016/ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)。M39有界收口；用户已[恢复RoboTwin排查且不改驱动](memory/global/requirements/decision-优先修复RoboTwin环境且保持系统驱动.md)，现原渲染恢复，无自动GPU后继。原始action保留原算子，当前算子审查聚焦B14。旧余量/历史计划不等于新授权。

## 证据与规则

[当前主张矩阵](experiments/single-layer-async-gemm-20261020/FOLLOWUP-CLAIM-EVIDENCE.md)：单层结构与有限数值可描述；H3新增调度收益/H4闭环有效成本未成立，完整1/3/7×cache强baseline矩阵缺。B14仍独立Q16实例，不是Q112序列。

engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission四项均false。原FAILED/retention/NO_GO/paused72/MLP0/22缺收据/default A_off保留。不能把target调用减少、成功子集或单block重放称E2E加速。

baseline及依赖冻结，只在optimized继续；旧投机代码/配置、原tile/HW、保护state/cycle/rules不改。BAAI写入仅~/cxz，AutoDL已归还不再连接；不改系统/驱动/时钟或杀他人。LIBERO全部对照同用优化reset，冷启动另列；RoboTwin0policy不声称新bank已验。

## 历史与发布

[Cycle完整历史](cycles/wam-research-20261016/history/ordered-followup-20261020/PROGRESS.md)、[更早M34历史](cycles/wam-research-20261016/history/M34-complete-20261020/PROGRESS.md)、[原知识清单](maintenance/hw-upgrade-20261016/LEGACY-RECORD-INVENTORY.json)仅按问题读取，不作当前权限或运行状态。

只维护[GitHub main](https://github.com/qingyiyi/WAM-egale/tree/main)作为核对过的当前源码/HW/小证据镜像，历史固定commit保留。权重、PT/KV/图像/DSO/工具链/私有SASS/连接配置不公开。HW固定v15不升级、不运行旧全量sync；兼容与回退见[维护报告](maintenance/hw-upgrade-20261016/README.md)。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00老师中文证据包；[官方投稿日期与profile缺项](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。正文仍由用户整理飞书，不能保证投稿/录用。
