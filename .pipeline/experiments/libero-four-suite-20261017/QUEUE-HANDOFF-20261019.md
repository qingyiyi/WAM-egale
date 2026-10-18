# 远端任务接续核验：已有自动队列

实测核验：北京时间2026-10-19 00:19:52。只读SSH返回0，未新启GPU/训练/episode，不改控制器或原协议。实际部署的Object controller、slot准入模块与BAAI campaign三源文件均与本地字节相同；无需新增SHA或smoke。

- GPU0是Goal的两个final worker，GPU1是Spatial的两个final worker；现场99%/100%，四个原final均未生成结束收据，没有已释放却未利用的槽。
- Object controller PID54413的出生身份/UID匹配，status心跳约16秒、无TERMINAL，实际状态WAITING_NATURALLY_FREED_REMOTE_SLOT_CURRENT14。
- 远端Object controller每30秒检查一次。任一个原final分片完整结束并通过清理检查、固定GPU空闲显存>=22000MiB且进程准入匹配后即可接入，不必等Goal和Spatial两个套件全部结束；每卡最多两个模型。
- 当前每个worker内部会自动运行下一完整episode，分片结束自动卸载/清理。Goal/Spatial campaign在各自final结束后自动生成CPU报告/终态。
- Object链已写好：剩14完整当前TF采集→整理全30条→固定ON3000微调→匹配Target200及T/C/P全部500组分片。部分final可与匹配Target阶段并行，首200组须等待匹配Target完成；不是人工逐项提交。
- 主提交者是远端controller；本机约5分钟CPU巡检仅检查存活/进度/异常，不负责逐局提交。用户暂不发消息并不会令远端控制器暂停。巡检实际26次、0错误。
- 失败、审计门或原截止会停止并保留PARTIAL，不盲目自动重试。异常需要Codex核对既有授权后处理；原10/21 16:56停新、17:11清理及所有质量false不变。既定链以外的新实验不会凭空进入队列。

最近完成数截至00:18:58：Goal276/500共同组、857/1500完整局；Spatial276/500、862/1500；Object剩14仍等槽。[派生摘要](QUEUE-HANDOFF-SUMMARY-20261019.json)。

代码：`投机采样/experiments/object_remote_resumption_v1/controller.py`的free_slots/run、`object_final_placement_v2/remote.py`的released_primary_slot、`tf_suite_pipeline_v1/campaign.py`的run。该结论是代码与实时进程/状态共同核验，不是仅凭计划称已启动下一任务。
