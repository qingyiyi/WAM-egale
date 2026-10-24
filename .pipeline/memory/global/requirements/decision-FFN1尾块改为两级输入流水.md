---
kind: decision
name: FFN1尾块改为两级输入流水
level: constraint
scope: cycle:wam-research-20261016;M39;FFN1尾块
status: active
updated: '2026-10-23'
supersedes:
  - 当前FFN1尾块固定三级输入缓冲的实现选择（原始版本与测量继续保留）
sources:
  - 用户解释三级A/B缓冲后要求“你把槽2去掉可以吗？这样shared mem是不是就少了”
---

# FFN1尾块改为两级输入流水

用户批准将当前研究算子的FFN1尾块从三级输入流水改为两级，以降低shared memory占用。只改该配置的缓冲级数，保留tile、K分段、warp分工、数学及BF16边界；不是删除每第三个K片段，也不改其他阶段的流水级数。

在当前L13研究worktree的optimized中修改并编译核对实际结构大小，原L13冻结副本与已测二进制保留。本次不推导已经提速或精度已通过；GPU正确性和性能需后续验证，不重开已闭合T0–T4窗口，不修改生产准入、旧账本及四项科学权限。baseline、投机侧和原tile不改。
