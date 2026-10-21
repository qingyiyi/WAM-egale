# RoboTwin 优先恢复与队列

更新时间：2026-10-21T11:44:03.604996+08:00。用户要求优先占一张远端GPU，当前已落实GPU1；GPU0诊断完成，正在CPU分析。

此前空闲原因：RoboTwin旧09:30窗口已结束，valid08单局补跑窗口不覆盖后继；算子前一测量阶段已终态。迁移没有完成，也没有隐藏运行的DEV队列。

已完成：Target48为41成功7自然失败；PARENT18为13成功5自然失败；CHILD2500/CONT3000固定训练、各10771行部署接口CAL完成。这些采集/校准不是最终配对质量结论。

11:32启动CURRENT18（六任务valid11/12 FIT、valid13 CAL）。当前快照完整终态1/18，成功1；active_case=stack_blocks_two-valid11。使用已冻结CONT3000与tau=0.007879490032792091；沿用108条完整512-token BF16 text bank，未改环境/原任务horizon。首局place_object_basket/valid11完整成功163.393秒。

队列：CURRENT18全部完整且cleanup闭合 → CPU监督审计 → 从固定CHILD2500执行ON3000的500 updates → CURRENT CAL6及Target CAL12部署接口核对。dispatcher已经真实启动并等待；没有GPU等待占用的重复记账。任一阶段失败即停止，不自动重跑机器人。

DEV90尚未启动/排队，仍需完成三臂评估入口核查与首两S审计准备；不把训练/校准完成等同迁移完成。四科学权限false，原失败/截止不动。

预算：新CURRENT作业≤7小时、每局≤1800秒/8000teacher、总≤144000teacher、清理预留900秒、证据≤2GiB。实际截止2026-10-21T18:32:41.501327+08:00。后继训练/CAL GPU生命周期≤2160秒、CPU审计≤1800秒、证据≤1GiB，最晚10/22 05:30退出；总额仍扣原24小时，启动前已用20744.747秒，余额约18.2376小时，不是新增加24小时。

CPU：server20/21、client16/17、控制18/19，与算子4/5错开。~/cxz/runs原证据约2.36GiB，未删旧失败。上一批18局约84分钟，CURRENT及训练/CAL常规估计约1.5–2.5小时；新状态可能出现失败长尾，此为估计不是完成承诺。

证据：[登记](REGISTRATION.json)、[固定18局](INPUT-MANIFEST.json)、[启动](LAUNCH.json)、[状态快照](STATUS.json)、[后继计划](../current18-followup-20261021/PLAN.json)、[后继启动](../current18-followup-20261021/LAUNCH.json)。远端实时状态：`/root/cxz/robotwin/runs/current18-20261021/STATUS.json`；后继`/root/cxz/robotwin/runs/current18-followup-20261021/STATUS.json`。
