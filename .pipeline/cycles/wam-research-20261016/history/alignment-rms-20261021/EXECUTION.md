# 当前执行检查点

## 2026-10-21T01:45:15.402937+08:00 M39 / M40

- 目的：先逐阶段对齐再调度，同时接续既定RoboTwin迁移。
- M39行动/结果：M64 attention接受；N256 FFN1有限精度通过但速度回退，已撤回；grid_constant已存在，取消0GPU。stage_final两个进程全部阶段有限精度通过；表见Progress。point8192正在构建/测量。
- M40行动/结果：Target48与PARENT2000齐；r3强化完整policy接口通过；parent剩余17继续，一局PARTIAL补跑待答复。未启动后续训练/DEV。
- 问题与影响：当前完整层仍约1.35–1.56倍native；调度deferred。固定parent FIT缺口阻止后续训练，不能自行改分区。
- 下一步：核point8192终态、量化剩余原因；持续巡检GPU1；更新HW/GitHub仅main。保护源与旧结果不改。
