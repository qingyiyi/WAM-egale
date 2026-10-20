# RoboTwin夜间阶段结果

更新2026-10-21T02:34:36.439687+08:00。Target48已齐（41成功/7自然失败），PARENT2000与一次CAL完成。
parent新访问采集现为17/18：13成功/4自然失败、259policy chunks、
12852target forward尝试/返回，GPU生命周期4582.732秒。
这是FIT/CAL采集，不是Target/TF配对质量比较或DEV成功率。

CPU审计终态PARTIAL_FIXED_PANEL_NO_TRAINING_PACK：
固定FIT仅11/12局，共5494行；CAL6/6，共3078行。
唯一缺口place_object_basket/valid08，原0动作重复terminal接口错误PARTIAL保留；
补跑请求仍待用户答复，没有自动重试、替换样本或缩小固定分区。
13个成功与4个自然失败均保留。后继CHILD/CONT/ON训练、current18及三臂DEV90未启动；
train_branch.py只完成CPU准备，不能记为已训练。

原03:55截止未重置，parent17在截止前完成。记录remaining_owned为空；服务SIGTERM返回-15，
不能伪造exit0，全部client退出0且完整终态已保存。两卡终态均无GPU计算进程。
CPU审计本次有真实PID+出生时间、返回码、回收收据；旧CPU审计缺出生记录不补造。
累计账本GPU生命周期5.6191/24小时。
根账本policy_episodes/teacher_forwards遗留聚合值为0，不能据此说没有运行；
当前数值来自各job/episode原记录，尚未改写根聚合字段。

输入/监督：实际到达root或已选parent的query；不是把全部候选树都当真实访问数据。
当前监督不保存重新query teacher所需完整视觉/KV条件，不进行不存在条件的递归重标注。
使用108条完整512-token BF16 prompt bank；本RoboTwin分支完整episode bank等价尚未测。
LIBERO四false、默认与旧失败不变；本结果不能签生产质量或性能准入。

证据：[17局终态](overnight-remote-evidence/parent-remaining17-20261021/STATUS.json)、
[固定18审计](overnight-remote-evidence/tf-migration-20261021/parent-audit-terminal17-r2/RESULT.json)、
[审计清理](overnight-remote-evidence/tf-migration-20261021/parent-audit-terminal17-r2/CLEANUP.json)。
