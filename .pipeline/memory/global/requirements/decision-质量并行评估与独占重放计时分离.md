---
kind: decision
name: 质量并行评估与独占重放计时分离
level: constraint
scope: project
status: active
updated: '2026-10-17T13:56:00+08:00'
supersedes: []
sources:
  - '用户要求非测速并行eval，同时保存中间状态，之后独占测试性能'
---

# 质量评估并行，真实状态重放与完整任务计时分别开展

非正式测速的完整任务成功率、环境步数和800失败罚分评估可使用获准的并行资源，保持每条episode的初态、环境、独立KV/RNG/scheduler与路由语义。当前优先两张卡各一个模型worker。用户进一步询问同卡多eval的可行性；这不是绕过显存或状态隔离核验、直接并发复用单个可变服务的授权。

质量运行同步保存有容量上限的代表性真实中间状态；记录其输入、完整必要条件/有效KV、RNG、实际候选和提交来源。优先少量短中长KV，禁止逐query复制全部历史。是否已具备完整replay条件以实际产物清单为准，只有actions/latents输出不算完整fixture。

之后在独占窗口重放，测量与实际保存边界相符的kernel、verifier或完整chunk成本。采集同步/序列化开销不得混入正式计时。冻结输入重放不能替代完整机器人episode，也不能推出真实任务E2E加速；后者仍需相同工程条件下无争用的完整任务比较。

并行运行中的墙钟保留为探索性吞吐和资源证据，不据此修改formal_performance_permission等原四项false，不以调用数或更快失败替代成本与质量。
