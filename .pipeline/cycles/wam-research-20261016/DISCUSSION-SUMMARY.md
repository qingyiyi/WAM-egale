---
kind: discussion-summary
cycle: wam-research-20261016
updated: 2026-10-16T00:06:11+08:00
---
# 研究约定摘要

10/17早用户已明确暂停实验，优先交付网页版draft结构/训练/开源文献Prompt，并同步GitHub。已在完整task5终态后停止，72/96、24缺项；GPU已清理。见[当前暂停决定](../../memory/global/requirements/decision-暂停TF实验并先外部文献复核.md)。旧10:00只是历史上限，不能作为自动恢复依据；只做必要CPU交接/发布，新网页方案仍须本地核验和新登记。

2026-10-17 01:35 用户将本轮夜间授权明确改为北京时间10/17 10:00，替代09:00；[Maintain决定](../../memory/global/requirements/decision-本轮TF验证截止与晨报时间.md)限定当前实验。旧登记不改，固定96清单/权重/质量门不变；现有DEV仍按原08:11硬帽，若需余量只做独立前瞻预算接续，未声称已延长活跃执行器。

2026-10-16夜间追加：完成当前1→2→3后，统一评估AutoDL单机双卡对采样/成功率收口与投机单独E2E测量的帮助，给出资源与工期；现在不单独回答。此要求不是新租卡或旧实例重连授权；正式性能仍按本机独占及原质量放行要求。原始资源观察已存当前实验master的FUTURE-DUAL-GPU-OBSERVATION.json，双卡实测吞吐UNKNOWN。

2026-10-16晚用户进一步明确：TF64确定为主draft，全力改善TF效果，MLP只作等容量baseline/ablation，不重开架构选择、不将MLP诊断置于TF改进之前；见[结构与投入决定](../../memory/global/requirements/decision-TF64固定为主模型与训练收口.md)。旧准入门和一次训练边界保留。

2026-10-16晚用户明确“我想优先把draft model搞定”：本阶段优先draft结构、训练方法与闭环可用性，暂缓新增纯算子性能工作；保留因果主线和原质量门，见[长期决定](../../memory/global/requirements/decision-当前阶段优先完成draft收口.md)。不把诊断假说写成用户决定。

用户目标是保成功与实际环境步数的WAM投机加速，并进一步研究verifier系统优化，面向CVPR2027。用户偏好Transformer，但现有TF只是研究候选；性能、质量和证据条件分开。

保留原tile代码/HW，不自动恢复撤回融合。中文稿在会话逐段讨论，由用户整理飞书，不新增本地论文稿。网页端完整方案已接收。核心是投机改变工作负载后，自研调度是否超越现有B14批处理收益。先真实shape/profile，再有限原型；普通batching不重新命名创新。用户授权夜间继续，AutoDL仅到10/16 09:00，累计4h链预算保留。

原始证据与授权范围见[当前完整报告](../../research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md)、[统计协议](../../research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md)和现有用户会话。此摘要不复制会话原文或凭据，也不是新实验授权。
