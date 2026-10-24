---
kind: decision
name: 分步实现FFN的TMA流水并按共享内存约束调整tile
level: constraint
scope: 当前5090单层算子FFN
status: active
updated: '2026-10-23T23:08:26.641496+08:00'
supersedes:
- preference-MegaMoE作为tile调度思想参考.md中的本轮仅讨论范围
sources:
- 用户：我同意你的设计方案，另外如果共享内存压力过大，也许tile大小也可以变小一点
---

用户批准按已讨论顺序实现：FFN2 TMA producer/consumer输入流水→单输出槽epilogue交接→FFN1/FFN2 K2048片段就绪。借鉴MegaMoE机制，适配SM120；不复刻SM100 UMMA/TMEM，不扩完整DAG。

允许共享内存压力过大时评估更小tile；这不是必须缩小或扩大尺寸搜索的要求。先核缓冲活跃期和编译资源，再用固定真实输入判断容量/任务数量/搬运/计算的代价。保留原M64及独立二进制，七段BF16次序、数值/状态门和原Torch/guarded三方eager对照不变。

本轮有界计划见experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/PLAN.md；原累计10GPU小时不增加，保护源码/旧失败/历史截止不修改。具体warp配置、tile及收益仍是待验证方案，不是采用事实。
