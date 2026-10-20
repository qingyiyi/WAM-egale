# 当前讨论与授权

最新用户明确同意按[两卡计划](ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)执行并完成汇报：[授权范围](../../memory/global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)。GPU1 RoboTwin有限迁移≤24生命周期h，GPU0算子≤8h，先原Target/数值再后继；旧账本/四false/保护项不变。


当前实际：O1/O2有限数值通过，O2仍慢native，O3有限FFN行组交接验证中；RoboTwin原渲染失败已止损，0policy/teacher。后续以PROGRESS与新报告为准；下列“只读/未授权”描述属于旧讨论，不覆盖顶部新授权。

此前用户要求先规划：远端一张GPU迁移投机侧到RoboTwin测试，同时规划算子下一步。[详细计划](ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)已交付，建议GPU1迁移、GPU0算子，尚未执行新作业；具体预算不是执行授权。[范围记忆](../../memory/global/requirements/decision-单卡规划RoboTwin迁移并行推进算子.md)替代仅泛泛调研的建议，原安装/训练边界保留。

此前已明确算子目标：整个单层Transformer在一个persistent kernel内，把全部算子切tile交给职责不同的warp，借鉴MegaMoE。只一层，不做30层大融合。baseline及共享依赖冻结，仅修改optimized；投机代码/配置与原tile不改。

用户允许四套LIBERO速度统一远端测，替代旧“最终E2E必须本机”的地点条款；每卡计时单worker且无观察到争用，M34 GPU验证等对应计时卡清理，CPU实现可并行。本机可用自然空闲窗口先核对精度；不杀他人。AutoDL已归还不再连接，BAAI所有写入限~/cxz。

当前M34结构实测成立但精度/性能NO-GO，已按600调用有界诊断收口；没有授权放宽原门。M35原360面板因Object1实施失败最多330，原失败保留、不自动重跑或延长截止；原未启动2/3可以按既定预算接续。RoboTwin仅调研，没有新增实验/安装/权重下载权限。

四质量权限保持false，旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off保留。不能用成功子集、调用减少、match-v1、单kernel结构或普通batch/cache宣称质量与协同加速。新训练/120/独立TEST/受阻45chunk不自动启动。

只main公开核对源码/HW/小证据，模型/PT/KV/图像/DSO/工具链/连接配置/凭据不公开。中文正文逐段交用户飞书，不建本地初稿。10/22 go/no-go、10/23冻结、10/28表、11/01 18:00老师中文证据包；官方投稿日期见日期报告，作者profile仍UNKNOWN。

旧讨论原文保留[归档](history/M34-complete-20261020/DISCUSSION-SUMMARY.md)；保护state/cycle/rules、不升级或旧sync。当前进度以PROGRESS与实际终态为准。

## 历史优先级（已由本轮执行进度接续）

用户明确先在自有单层kernel内部实现FA2，再研究跨阶段同步；要求量化误差并调查RoboTwin/LIBERO差异。已完成内部self/cross/split-combine与18同输入有限逐位核验，整层仍数值NO-GO。优先处理FFN2/Q与少量Norm2误差；不因attention局部PASS签发完整block/闭环质量证书。

直接在optimized迭代、不为另一agent保留额外代码版本、不另建登记；沿M39保留原始测量与失败。新阶段387block/216计时，GPU进程已清理，未启动后继优化或机器人。当前仍全阶段grid barrier，attention的4计算warp自己负责load/MMA/softmax/store，尚无跨阶段ready队列。

RoboTwin只读核验：50双臂任务，Easy/Hard不是新增task；官方Target存在，draft需适配32token/16有效通道并重训/校准。只提出迁移评估，不安装环境、下载权重或启动15000局。当前四false不变。
