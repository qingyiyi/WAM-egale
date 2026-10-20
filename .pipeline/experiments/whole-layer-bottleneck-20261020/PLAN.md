---
kind: experiment
id: M38
cycle: wam-research-20261016
status: registered
---
# 当前整层算子瓶颈诊断

授权：用户最新明确将主要精力转到“讲解当前实现并 profile 为什么慢”。本轮可诊断未准入原型，不要求先解决逐位差异；不降低数值/生产门。

输入为既有真实 action block0，B2/B14 × KV288/1152/2160，共6签名。固定 whole_layer.cu SHA256=df8b386f302b9904ac4ef8ab08d171ef6d9e6f8143451831c3318f5e0835ef0e；不改 baseline、投机代码或原实现。参考为原Torch完整block及已部署 sequential NativeBlock；比较双方未缓存 cross K/V 的完整block，不冒充最强 cached/Graph。

1. 同输入核对原Torch/native，原whole/旧阶段计时副本/新静态分配副本的输出、中间量、只读有效KV、输入与RNG。新副本只把每阶段全局 atomicAdd 任务领取换成 CTA 固定 grid-stride，保留所有数学、barrier和stage计时；唯一调度反事实，不扫参数。
2. 六签名，各路径一次预热、六轮交错 CUDA event 计时；profile采集与计时分离。原whole阶段globaltimer取三次；原Torch Linear用完全相同的whole中间输入做10类GEMM组件测量，记录真实M/N/K、kernel和GPU时间。
3. 对照native与whole kernel时间线、GEMM组件、阶段分解及静态任务领取，量化慢点。理论tile利用率/资源上限与实测指标分列。NCU已知权限失败，不提权；DRAM/L2/实际occupancy/stall counters仍UNKNOWN。

远端GPU0一worker，CPU4/5两核；启动前确认无其他context。新生命周期总帽2700秒，T0+2400停工作、+2700清理；构建≤900秒、worker≤900秒。完整block≤240，独立GEMM≤600，0机器人/30层/训练，新增≤512MiB。等待、构建、失败和清理计预算，旧M34/M37预算不重置。产物就在本namespace与远端/root/cxz/ops/whole-layer-bottleneck-20261020；不写~/cxz以外。

每项attempt保留输入/源码身份、开始/终态、PID+出生ticks/PGID/UID、退出方式。新副本若改变原whole输出/状态，停止该副本后续计时，不将实施失败叫优化无效。出现外来context停止新计时，不杀他人。硬截止不重试。

交付：RESULT.md/json、逐样本计时、阶段/组件CSV、原始trace、资源与清理报告、源码讲解、HW/main的小证据索引。四质量字段false及全部旧FAILED保持。
