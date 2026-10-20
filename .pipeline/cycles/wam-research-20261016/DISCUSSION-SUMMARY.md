# 当前讨论与授权

用户最新要求：RoboTwin继续；算子先分析当前剩余开销、解释具体实现并推荐下一步。[确认决定](../../memory/global/requirements/decision-接续RoboTwin并先分析当前算子开销.md)。

当前context128数学源码未改；24完整block和150原语profile完成。普通速度仍上一轮B14三KV约1.73–1.88倍native。阶段grid.sync仍在，8warp协作epilogue不等于完整异步；NCU计数权限被拒，动态spill/occupancy未知。最新反馈改为先逐阶段三路径对齐，再修主要差距，最后context K/V ready，保持数值门。

RoboTwin最新明确接续允许固定未开始42条Target进入新namespace，不重跑旧失败，不重置原24GPU小时总账。GPU1固定队列自动接续；现场10/20 21:59:00北京：新增5/42完整终态、5成功，当前`open_microwave-valid01`；原首六5成功1失败保留。新阶段截止次日03:37:02，6小时含清理；TF训练尚未启动，Target采集不冒充TF评测。

仅单层persistent kernel；baseline/投机/原tile/保护HW不动。四质量权限false，旧FAILED/retention/paused72/MLP0/22缺收据/default A_off不变。BAAI只写~/cxz，AutoDL已归还。不改驱动/服务/时钟，不杀他人。main只同步源码/HW/小证据。

此前[讨论](history/context128-profile-start-20261020/DISCUSSION-SUMMARY.md)保留。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包不变。

最新[逐阶段对齐计划](../../experiments/single-layer-async-gemm-20261020/context128-profile/STAGE-ALIGNMENT-PLAN.md)尚未启动新GPU；当前profile预算已终态。
