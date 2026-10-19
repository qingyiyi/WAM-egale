# 最近执行检查点

2026-10-19T10:06:11.045625+08:00：完成当前算子源码导读/后续清单，Maintain保存当前交付范围要求。见[代码与计划](../../../投机采样/docs/OPERATOR-CODE-AND-NEXT-STEPS-20261019.md)。本次无新GPU作业。

## 2026-10-19T09:55:55.629844+08:00：按最新要求先交付直接Torch复制版

三个完整定义逐字/AST一致，可撤销替换保留参数/cache地址；六真实B2/B14短中长fixture的18次GPU数值/有效状态/输入/RNG及历史gold逐位通过。294完整block、约15秒worker自然退出；六点中位耗时比0.9677–1.0033，单轮波动不当作加速。一次supervisor相对import在GPU前失败并保留，修正未改数学。总412/5000调用，原截止不重置。旧66/旧A/A失败保留，旧多引擎watch退出、fixed3 AA未跑。完整模型安装入口已准备，但30层/机器人未由本轮新增证明。[实际结果](../../experiments/direct-torch-block-copy-20261019/RESULT.md)。

## 2026-10-19T09:34:57.532780+08:00：Object14完整与训练/评估实际接续

14/14补采自然完整、原数据准入通过（FIT6164/CAL3080、无反事实/额外teacher），guard=CLEAN。ON3000固定训练已完成，全500阶段启动；09:21 T20018/200、首FINAL5/225局，共同完成组1个（以09:21快照为准）。每卡一worker；额外并发因NVML/容器PID未证明对应继续拦截，不假称自有。原任务继续，无新重试；[完整实证](../../experiments/libero-four-suite-morning-20261019/RESULT.md)。

## 2026-10-19T09:09:48.963521+08:00：今晨GPU与两机接续

M24新授权后66次GPU完整block严格数值/有效KV/输入/RNG及旧gold通过；A/A三轮最大2.512635%超2.5%，停止计时且保留原始失败。118 block，两worker自然退出；CPU监控等待资源真正变化，原10:48:46截止不重置。[新结果](../../experiments/wam-block-gpu-followup-20261019/RESULT.md)。

BAAI Goal/Spatial三臂各500全齐，成功T/C/P为463/464/467、472/466/464；全部成败/步数/800罚分/配对胜负已CPU独立复核。Object因capture编号碰撞新14监督无效，旧16有效；用户明确批准补采固定14，新v3独立namespace已在两卡运行CURRENT14和匹配Target200。v2启动前遇过期T_POOL模板，无GPU/episode，旧失败保留；新Matched使用原M17截止而非延长旧采集。新health-v2第一巡检PID出生匹配、两GPU已占用。

Maintain保存本机空闲接续及Object固定14补采两项约束；新巡检成功后旧CPU watcher收到带原因的停止标志，无远端信号/研究作业终止。HW Progress改为五列支持状态，科学门另列，不借completed宣称质量通过。main已发布并核对2a91ce51f4cc39e99495e0014d01b145349b4324，小源/派生结果公开，权重/tensor/大原报告未上传。

更早检查点完整见[归档](history/M24-before-GPU-followup-20261019/EXECUTION.md)。
