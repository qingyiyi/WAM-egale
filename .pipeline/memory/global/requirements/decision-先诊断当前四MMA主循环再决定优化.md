---
kind: decision
name: 先诊断当前四MMA主循环再决定优化
level: constraint
scope: cycle:wam-research-20261016;M39-current-projection-diagnostic
status: active
updated: '2026-10-24'
sources:
  - 用户在result_empty诊断后明确同意先按建议拆解当前四MMA主循环
supersedes: []
---
# 先诊断当前四MMA主循环，再决定下一项改动

本次执行范围为当前三个小投影的输入就绪等待、操作数加载/转换及MMA主循环软件观察。使用独立诊断副本，保留当前四MMA、一TMA、三epilogue分工、tile、K顺序及BF16边界；不自动增加warp或实现TMA输出写回。

沿原累计十小时余量，具体短窗口和停止条件写入现有M39实验。先核数值/有效状态与插桩扰动，再解释分量；旧八MMA数据不能直接替代当前测量。任务结束打印并保存完整阶段表，未重测的阶段明确历史。具体假设与结果只保存在实验，不作为长期规则。
