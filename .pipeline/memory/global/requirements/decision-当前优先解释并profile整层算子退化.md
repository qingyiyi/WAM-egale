---
kind: decision
name: 当前优先解释并profile整层算子退化
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
sources:
  - 用户：暂时先这样，现在主要精力放在算子为什么这么慢，讲解实现并profile
supersedes:
  - decision-继续数值契约修复后验证单层异步tile流水.md中必须先完成数值修复才推进后续诊断的执行优先级
---
# 当前优先解释实现与测量慢点

先围绕当前单层完整persistent kernel开展有界真实输入profile、与原Torch参考的成本对照和必要隔离反事实，输出源代码讲解与实测瓶颈。可以测量未准入原型的性能问题，但不能将其称为已对齐的实现。

只改变当前研究顺序。逐位/状态/闭环门、baseline及依赖冻结、投机侧和原tile不改、远端只~/cxz、四权限false与新作业预算/清理要求全部保留。停止数值盲扫，不扩大训练或机器人评测。本次未改变成功率/成本报告口径。
