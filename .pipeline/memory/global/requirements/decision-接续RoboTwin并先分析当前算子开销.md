---
kind: decision
name: 接续RoboTwin并先分析当前算子开销
level: constraint
scope: cycle:wam-research-20261016;M39;M40
status: active
sources:
  - 用户2026-10-20明确要求RoboTwin先继续，算子先分析当前版本剩余开销、结合具体实现讲解并建议后续优化
supersedes:
  - decision-继续实现并评估更大tile.md中的本次RoboTwin仅关注、不启动后继条款
---
# 接续RoboTwin，先解释当前算子剩余开销

用户明确接续既定RoboTwin有限迁移，不再仅监看首六局终态；可在原六任务/174局范围和24GPU生命周期小时总帽内，为未开始后继阶段建立新的有界登记，保留原截止、失败与消费账本，不把旧已过期作业原地重启。原Target首六FIT已结束，后续从尚未执行的固定任务/初态继续；不扩大到50任务或新训练搜索。

算子优先对当前context128版本实际profile并结合源码解释warp职责、tile、数据依赖与同步，再按证据安排后续优化。不能将旧版本profile占比当成当前事实；不在本次分析中擅改数值标准或冻结baseline。用户要求的是具体实现和有证据的优化顺序，不是只给泛泛计划。
