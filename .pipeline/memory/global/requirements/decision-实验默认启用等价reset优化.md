---
kind: decision
name: 实验默认启用等价reset优化
level: constraint
scope: project
status: active
updated: '2026-10-17T11:26:47.645224+08:00'
supersedes:
  - '性能主题记忆中未来实验可以选择启用优化reset的可选措辞'
sources:
  - '2026-10-17用户明确要求写入HW，以后实验都用优化版本以提高测试效率'
---

# 后续实验默认使用已验证等价的reset优化

项目后续涉及policy reset的采样、完整任务成功率验证及性能实验，默认启用已完成完整任务等价验证的reset优化，保持原512-token BF16 text bank；所有对照臂使用同一有效bank与相同工程条件。不是只对投机一侧启用，也不使用此前裁padding失败版本。

首次bank构建、模型装载及冷启动成本单列，warm任务成本注明边界；不能以缓存普通收益冒充新MegaTransformer贡献或未测E2E收益。专门研究reset时可显式登记未优化对照，不把它设为日常实验默认。

bank的target身份、提示词、token长度/dtype与当前配置必须匹配；不匹配不能静默复用旧bank或悄悄换回慢路径，先如实报告具体问题。当前AutoDL无SHA/独立测试要求继续有效：独立no-hash入口保留原get/install数学与原bank，必要一致性记录随完整任务，不伪造未执行门通过。原始等价证据及失败版本保留。

来源与数值：[性能/reset主题](../knowledge/decision-性能证据与reset使用.md)，本要求不改变原质量门、预算或算法。训练不涉及reset的部分不额外引入bank。
