# 当前讨论与授权

用户要求继续实现算子、允许tile切大，并继续关注RoboTwin。[用户选择](../../memory/global/requirements/decision-继续实现并评估更大tile.md)保持约束级、M39范围。只在optimized迭代，baseline/投机/原tile不动；整个单层一个persistent kernel，不改成30层。

新有界两variant完成，context128通过有限精度并在B14三KV均小于同轮native两倍，作为当前研究实现。仍慢约1.73–1.88倍，不是Torch等速或机器人加速。actionM224保持64行；8warp协作epilogue和2048point任务不等于完整异步DAG。阶段grid.sync仍在；新ready未验，默认关闭。

新预算372/420完整block、GPU生命周期69.379/900秒，原旧1497/1500、旧资源FAILED和20:20:28截止未重置；无GPU后继队列。下一阶段先当前瓶颈，再真实依赖ready调度。

RoboTwin原Target首六FIT 5成功1失败，旧0动作PARTIAL保留。TF collector/训练/DEV待实现，不宣称自动排队；本轮只关注，旧截止不续期。BAAI只写~/cxz；AutoDL归还；不改驱动/服务/时钟、不杀他人。

四质量权限false；旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保持。仅main发布源码/HW/小证据，不发布模型/PT/DSO/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包不变。此前[讨论](history/larger-tile-followup-start-20261020/DISCUSSION-SUMMARY.md)保留。
