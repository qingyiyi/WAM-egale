---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;experiment:sm120-ffn-next-20261024
status: active
updated: '2026-10-24'
sources:
  - 用户明确读取并执行WAM_SM120_FFN_NEXT_CODEX_PROMPT_20261024.md
---
# SM120七段FFN先诊断提交成本，再有限合并owner

本轮冻结当前实际保留SM120 A的独立源码及二进制，不回旧A100319ac6a。先核七个K2048的精确BF16递推、旧负结果和当前成本；相邻两段同CTA仅为证据支持后可试的一个候选，不要求去掉global ACT。最多一个设计、一次有明确定位修订，不重做已充分排除的专职分工/M16驻留/DSM版本。

保留FFN1完整归约、GELU前后舍入、FFN2逐段初始化/FP32加法/七次BF16舍入、末段bias/门控残差顺序。精度先于正常完整层计时，局部改善不替代整层收益。每个T0–T3和独立候选/失败/回退结束，聊天直接打印完整当前阶段表并保存快照；未重测标历史或未测，评估与实际保留身份分列。

沿用现有5090资源授权与历史账本，每job有限。科学门、原失败、投机/baseline/原tile/protected源和双A100预算不变；无证据空间可以暂缓收口，不以实现数量作为完成标准。
