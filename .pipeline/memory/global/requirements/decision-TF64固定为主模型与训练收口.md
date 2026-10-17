---
kind: decision
name: TF64固定为主模型与训练收口
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-16T19:50:00+08:00'
supersedes:
  - '当前阶段重新在TF与MLP之间选择主draft的解释'
  - '将新增MLP风险诊断排在TF效果改进之前的建议'
sources:
  - '2026-10-16用户：按照网页端的要求，我们不是还是确定tf作为draft model吗，全力搞定tf的效果'
  - '2026-10-16用户：我想优先把draft model搞定'
---

# TF64 固定为当前主模型，集中训练与闭环效果

当前研究主 draft 确定为 TF64 Transformer，不重新进行架构选择。优先改善 TF 的权重、训练分布和完整闭环效果；MLP 仅作同容量论文 baseline/ablation，不作为当前主模型替换路线，不优先投入新MLP调参或诊断。

这固定研究结构与资源优先级，不等于当前 CONT3000 权重/配方通过验收。旧 FAILED、retention、原阈值、全局A_off与四项false权限保留。成功率/长度及TF相对matched MLP的既有门不因结构偏好放宽。

先诊断再训练、一次新FIT/CAL和等预算训练上限继续有效；集中改善TF不是无限架构/种子/阈值搜索授权。与TF效果直接相关的数值与数据排查允许继续，纯算子性能原型后置。MLP的必要公平对照在TF候选形成后补齐。

本决定细化[优先完成draft收口](decision-当前阶段优先完成draft收口.md)；该记录的实验边界仍有效。临时数值诊断及因果假说只保存在实验记录，不提升为用户决定。
