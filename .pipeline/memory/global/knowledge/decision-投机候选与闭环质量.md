---
kind: decision
name: 投机候选与闭环质量
level: reference
scope: research
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/speculative-sampling/RESUMED-EXPERIMENTS-AND-CLEANUP-RESULT-20261015.md
- .pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md
- 投机采样/docs/CODE-MAP-20261014.md
---

# 投机候选与闭环质量

截至10/15本机终态：TF64为16个action token、17维输入、1层pre-norm Encoder、d64/4heads/FFN256/GELU/dropout0、两proposal heads，53070参数；无图像或prompt cross attention。MLP96为52928参数对照。

fresh30同批Long10/init24：T/MLP child/TF child成功10/9/9，draft均task7失败；800失败罚分均值272.1/332.0/311.2。TF成功配对步数比P50/P90/max=.9956/1.0415/1.1622，长度条件通过但成功数门未过。

真实同seed TF23 parent2000为9/10、失败task1、罚分333.2；child2500为9/10、失败task7、罚分311.2，下降6.60%。父子全10初始像素/物理/RNG逐位。预登记成功数→罚分→并列parent规则选择child仅供扩大研究，`quality_gate_passed=false`。全局仍A_off，新TF正式速度未跑。

原fresh30因task8初始腕相机4–5像素差异FAILED，原失败保留。另有物理/环境/target RNG全配对的独立统计诊断；不能冒充原严格观察配对PASS。开发10局不是独立TEST或非劣保证。

结构研究骨架可暂定，训练/接受配方和闭环质量未收尾。原40与新30不能混挑最佳结果，7406棵树规则无违规不等于未保存velocity全量复算。
