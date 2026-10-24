# 显存争用中断后的未开始队列接续

2026-10-25 00:53，pair406 的 Target 在观察 VAE 编码时 CUDA OOM，申请1.32GiB、可用271.69MiB；错误同时列出其他显存上下文。直接原因已观察到，其他上下文的用户身份未知。原 C/P 已自然失败至400命令，仍是有效算法失败，绝不删除或改成基础设施失败。

- [x] 读取原 HALT、两 worker 清理、显存与错误原件，保存在 BEFORE.json；未修改旧实验。
- [x] 同原 GPU 新 namespace 接续未领取完整组，冻结数学、模型、reset bank、种子及 pair 审计。只增加资源准入：加载前空闲≥24GiB、每个完整 episode 开始前≥6GiB；不杀其他进程。
- [x] 所有等待/准备/失败/清理继续记原160 GPU小时账本，原T0及 guard 更早硬截止不改；同一 pair 三臂同卡。每卡一个模型 worker。
- [x] 新 CPU watch 继续原唯一队列及已授权 pair281/C 的一次尾部补齐。
- [x] pair406/T 不自动重试，原 PARTIAL 与已完成 C/P 保留；没有它不能报告1800完整或600完整组。

用户2026-10-24明确要求两卡持续接续直至原固定评估完整，覆盖本次未领取队列恢复；不将该要求解释为取消原禁止自动重试、扩大样本、160小时预算或改变科学门。若资源持续争用，则如实记资源阻塞，不伪造并发或承诺ETA。

输入为原 INPUT-MANIFEST.json（600组/1800臂），新 worker 的 client/server/route/engineering/device binding 字节不变；仅 guard 的显存准入与新作业路径不同。新增轻量代码/日志≤64MiB，仍受原12GiB结果cap。终态/资源/新PID与出生身份/待补项写 RECEIPT.json、PROGRESS.json；原四false保留。

现场核验：2026-10-25T01:01:57.546414+08:00两worker PID/出生身份存活，均RUNNING_COMPLETE_DEV_EPISODE，HALT=null；405完整组三臂，GPU1新guard1283071/birth745972217。RoboTwin累计约86.05/160GPU小时；新watch每60秒核查且保留原pair281尾部接续。详见PROGRESS.json、RESUME-STDOUT.json。
