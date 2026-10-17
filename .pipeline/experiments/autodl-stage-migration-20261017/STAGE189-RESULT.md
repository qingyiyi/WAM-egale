# Long阶段189组三臂：描述结果

固定暂停清单全部纳入。不是500组全量完成、独立TEST、质量通过或正式性能结果；旧FAILED和初态配对缺口保留。

|方法|成功/完整局|实际命令步均值|800失败罚分均值|单局墙钟P50/P90秒（并行探索）|
|---|---|---|---|---|
|Target|183/189|284.69|284.81|180.9/277.9|
|TF连续|185/189|274.67|274.76|115.3/168.2|
|TF＋PF2|181/189|284.05|284.22|137.3/207.0|

ΣT_round/Σg仅在全部round计时覆盖时计算，缺失标UNKNOWN。调用次数不代表延迟；共享并行墙钟不能作为正式E2E加速。

逐局及成对成败/长度/任务分层见RESULT.json、EPISODES.csv。当前四项权限全部false。

原件：[统计JSON](../../../投机采样/results/runtime/p0-long-stage189-descriptive-local-cpu1-v1-20261017-r1/RESULT.json)、[逐局CSV](../../../投机采样/results/runtime/p0-long-stage189-descriptive-local-cpu1-v1-20261017-r1/EPISODES.csv)。相对Target共同成功最大长度比C1.7004、P1.5465，旧长度工程条件仍不能宣称全部通过；C/P组件审计FAILED分别15/27保留。四权限false，旧NO_GO不改。

备份：p0-autodl-migration-backup-local-cpu1-v2-20261017-r2，37349文件/1970600995B原payload，tar2053969920B，650.704秒，exit0/CLEAN、源删除0、无SHA。后续新job结果不在该冻结快照内，迁移前另存增量。
