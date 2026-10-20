---
kind: decision
name: 继续数值契约修复后验证单层异步tile流水
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
updated: '2026-10-20T09:24:01.490859+08:00'
supersedes:
  - 当前Cycle在M34收口后仅讨论、不继续新算子研究的等待状态
sources:
  - 用户最新明确：可以的我同意你继续做，另外告诉我为什么tf的加速比在long和object上这么低
---
# 继续有限数值契约研究，再验证异步tile流水

用户同意承接上次交付的下一步：先处理真实输入上Q、FFN2与attention设备数值边界，再在成立的原语上推进一个单层异步tile流水。仅optimized独立版本，baseline及共享依赖、投机源码、原tile和历史证据冻结；只做一层，不恢复30层或新增机器人实验。

本次可在现有BAAI资源中开展必要的有界源码实现与验证，启动前核实资源并登记新预算、输入、清理与终态；不能继承M34已耗尽600调用或M35关闭的时间窗口。数值门失败时保留结果，不把match失败原型、普通allclose或单kernel结构当作准入。没有扩大四项质量权限，也没有新增训练、TEST、租卡或第二环境授权。

用户同时要求解释Long/Object加速比；这属于基于已有完整结果的成本分析，不能为了改善倍数删除失败或慢成功。此前仅讨论的等待状态被本次继续授权替代，已关闭实验原终态不改。
