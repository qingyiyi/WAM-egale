---
kind: requirement
level: constraint
scope: cycle:wam-research-20261016;experiment:local-warp-pipeline-20261023
status: active
updated: '2026-10-23'
sources:
  - 用户要求读取并执行WAM_LOCAL_WARP_PIPELINE_CODEX_PROMPT_20261023.md
supersedes:
  - 当前算子暂停等待网页设计（仅本轮有界T0至T4）
---
# 局部warp流水每项结束输出完整阶段表

用户授权按T0至T4自主执行本轮，保留独立A源码/二进制。主线仅FFN2现有四warp MMA下其余四warp准备同tile数据；副线独立LN1调制，满足条件后LN3。GPU本轮最多120分钟、原累计剩余更小时取较小者，48小时硬收口，不重置旧账。

每个任务与独立候选完成、失败、回退、提前停止，都必须在聊天直接打印完整block结构表、计时状态和完整层比较并保存快照；不能只给链接/改动行。注明评估变体与实际保留版本，未重测标历史或未测。采用由严格正确性及三KV无插桩完整层结果决定，不预设正结果，不扩展DAG/统一warp比例/FA2重写。范围内不重复索取确认。

本轮附件明确要求A/D/G同轮强基线比较；这是仅本轮对旧“只与原Torch比较”展示偏好的具体覆盖，不删除旧记录。执行于10/23完成，结果入口见local-warp实验；后续不把本次有限授权扩成新方向。
