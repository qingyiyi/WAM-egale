---
kind: decision
name: 自有整层读取预计算CrossKV并对齐逐阶段耗时
level: constraint
scope: wam-research-20261016/M39
status: active
sources:
  - 用户明确要求自有算子也读预计算并对比性能，输出每部分耗时的完整阶段表
supersedes: []
---
# 自有整层读取预计算 Cross K/V，并对齐逐阶段耗时

在当前optimized分组复用实现上，接入每层每条件组预计算K（含原RMS）和V。双方使用同等缓存边界，先有限真实输入精度/状态验证，再比较短中长KV完整层，并给逐阶段计算与同步等待分析。冷构建和warm调用分列，profile不冒充正常计时。

本轮不改变投机、baseline源代码、原tile、保护HW、数值门或科学权限。沿原累计十小时预算登记新的有界阶段；旧截止/结果保留。保持RMS数学与BF16边界，不趁机增加tile+RMS融合或ready调度。
