# Long全量实际运行检查点

检查时间：2026-10-17 15:21:18，北京。此文件是运行检查点，不是完整质量结果或通过证书。

- OBSERVED：旧Long v1 r1收尾23组三臂、69完整episode、72593次target尝试。原监督终态`PARTIAL_OR_FAILED_LONG500_COVERAGE`保留；自有组清理`OWNED_GROUPS_EMPTY`，不是将旧作业改为完整500通过。
- OBSERVED：第一次v2 r1控制尝试等不到共同最后臂窗口，`NO_COMMON_LAST_ARM_WINDOW_NO_CHANGE`，480.5秒、未接管/未启动新GPU。独立r2控制只暂停出生身份匹配的CPU runner，当前episode自然结束，再用原入口补完同case剩余已授权臂；没有截断或重跑。控制383.465秒，真实wait rc0及清理收据落盘。
- OBSERVED：新v2 r2 runner28232、guard28231；四server在15:14:44–45实际READY，15:14:51开始原完整任务。GPU0/1各两独立模型/环境/KV/RNG，server/client各2线程，总16CPU。15:18两卡利用率98%/97%、显存28791/28917MiB；不是正式性能数据。
- OBSERVED：15:21新9局自然终态、1组三臂已闭合，四worker累计实际target尝试2440/2321/2134/2134，计数随运行增长。旧69局与新结果来源分列，不据少量结果宣布TF合格。
- UNKNOWN：v1 optional capture曾重复字段TypeError；v2修复后又出现`actual candidate action KV bucket differs`。原gap保留，288桶未捕获、中长未尝试，尚无合格的新TF完整replay fixture。质量episode继续，不能声称保存成功。

源码：科学父`4f04b35e44cbc00483dc6245ad57abec718385c9`；旧v1执行`67d1c0d0f5c9b25ecfb512dcdc71aebf5cfc9cca`；当前四worker执行`d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9`；独立handoff控制`34690997f1cdbb54a4beaf4012e6991c5c736e17`。控制Git HEAD不冒充模型执行版本。

新plan只读复用原23组，477组/1431局待新worker完成；原48h起点10/17 13:55:56、工作停10/19 13:40:56、清理截止13:55:56，系列6,001,000 target尝试/10GiB（fixture≤3GiB）/磁盘空闲≥5GiB不重置。两并发/四并发时段成本分层；不用共享卡时间签正式E2E。所有臂同完整512-token BF16优化reset bank。无重试、无新Long训练、无90任务、无120/独立TEST/正式性能。

四权限均false，旧TF_ON/PF2 NO_GO、FAILED、retention/default及保护文件不改。现有50初态含暴露FIT/DEV，描述性覆盖不冒称独立TEST。

证据（本地镜像按需继续回传）：
- [当前原plan](../../../投机采样/results/runtime/p0-tf-fullsuite-long-v2-20261017-r2/plan.json)
- [实际接续结果](../../../投机采样/results/runtime/p0-tf-fullsuite-long-v2-20261017-r2/TRANSITION-RESULT.json)
- [控制器真实清理](../../../投机采样/results/runtime/p0-tf-fullsuite-long-v2-20261017-r2/CONTROL-OWNED-CLEANUP.json)
- [15:21现场观察](../../../投机采样/results/runtime/p0-tf-fullsuite-long-v2-20261017-r2/observations/ACTUAL-20261017-1524-r2.json)；文件标签不是执行时间，以内部at为准。r1观察误用EPISODE.json查询、零计数不可用，原件保留，r2按实际EPISODE-TERMINAL.json更正。
- [评测范围依据](SCOPE-REVIEW.md)

NEXT：监督完整任务与预算/自有清理，回传旧69原件；修复fixture需要独立代码版本和自然边界接入，不改运行中冻结数学路径。最终自然成功/失败、800罚分、实际步数与三组配对联合汇总。实例尚未关闭；退出SSH不等于停止计费。
