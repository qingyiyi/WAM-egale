---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-19T14:49:18.383103+08:00'
---
# 最近执行检查点

## 2026-10-19T14:49:18.383103+08:00：短KV/B14核查与比较方向澄清

用户确认关注自有实现比Torch慢；继续询问2.5%含义和源码/完整Transformer覆盖。M27分列默认复现/Torch与FFN融合/同配置未融合；后者约3%为更快。源码绝对路径、默认仅SDPA及可选FFN1边界已解释并通过Maintain合并进可读记忆。

新增A/A真实重放两次foreign GPU中断、一次有限等待到期，无自动扩展；665.685秒生命周期含等待/失败/清理，GPU阶段墙钟31.145秒非CUDA active，6180attention/918block intents，0full30/机器人episode。三个owned实际退出，无他人进程被清理，旧PARTIAL及所有样本保留。短/中B14同等cached Graph比约0.999/1.001，不把未全矩阵包装通过。

只增加共享诊断工具、CPU预读参数及文档；baseline、1875投机文件、保护文件/原tile不改。未找到稳定的新增数学开销证据；未盲改CUDA内核。旧M22慢点、M25严格筛查、M26完整FFN2失败及NCU偏差仍在。[完整结果](../../experiments/operator-shortkv-b14-20261019/RESULT.md)。

原Object巡检06:46:38UTC：{'T': 426, 'C': 227, 'P': 227}，3活动/自动后继，无terminal；原M17预算不变。

原执行归档：[历史](history/M27-before-investigation-20261019/EXECUTION.md)。
