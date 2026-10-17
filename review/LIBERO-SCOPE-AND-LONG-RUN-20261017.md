# LIBERO范围建议与Long全量运行

截至2026-10-17 15:21北京时间。这是后续获授权工作的运行检查点，非完整实验结果。固定科学快照`4f04b35e44cbc00483dc6245ad57abec718385c9`的paused72身份永久保留；不以本分支或main覆盖旧结论。

## 范围核验与建议

LIBERO-90不是当前论文主线的通用必测项。[LIBERO官方](https://github.com/Lifelong-Robot-Learning/LIBERO)将LIBERO-100拆为90预训练和10下游终身学习评测。现代VLA论文常用的Spatial/Object/Goal/Long四套各10任务，也不等于LIBERO100。

[CoT-VLA（CVPR2025）作者说明](https://cot-vla.github.io/)使用四套、每套500初态三轮；[LingBot-VA原文§4.3.2/Table2](https://arxiv.org/html/2601.21998v1)同四套，每套500局×3随机种子；[FLASH预印本§4.1](https://arxiv.org/html/2605.13778v1)亦四套且每套独立draft。[Seer（ICLR2025 Oral）官方](https://github.com/InternRobotics/Seer)的LIBERO部分聚焦Long，另有CALVIN和真机，不能推出Long唯一实验即可保证论文充分性。

建议先收口Long Target/冻结TF_ON连续/同权重PF2，报告成功、800失败罚分、实际长度、有效提交成本、完整chunk与任务E2E。单套件不能支持跨WAM/环境普适性；主线成立且有预算后，再考虑一个有可信Target的10任务套件，而非默认扩90。当前用户询问仍记为范围研究，未冒记为取消旧授权；90因无可信同代Target持续BLOCKED，未训练或采集。

按50初态×三臂，Long为1500局，90另13500局；省去90意味着减少原Long+90计划90%的episode数，不保证同比墙钟节省。

## 已观察运行状态与缺口

Long500组计划已获新授权，含暴露FIT/DEV初态，为描述覆盖而非独立TEST。原v1收尾23组三臂/69完整局、72593 forward，原PARTIAL及清理EMPTY保留。新v2四worker已实际READY并运行（两GPU各2独立模型/环境/KV/RNG，总16CPU），15:21新增9自然终态、1组三臂闭合。无截断或episode重试；原48h绝对起止、系列forward/存储预算不重置。

科学父4f04；旧v1执行`67d1c0d0f5c9b25ecfb512dcdc71aebf5cfc9cca`；当前四worker执行`d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9`；独立交接控制`34690997f1cdbb54a4beaf4012e6991c5c736e17`。报告HEAD不冒充模型执行源。

全臂同完整512-token BF16优化reset bank。并发耗时仅探索性资源数据，正式性能另需独占与质量许可。v1 optional capture重复日志字段失败；v2修该项后真实query又遇KV桶检查gap，仍UNKNOWN，尚无合格新TF完整replay。旧失败均保留。

`engineering_quality_screen=false`、`independent_test_complete=false`、`noninferiority_established=false`、`formal_performance_permission=false`。旧TF_ON/PF2 NO_GO、retention、threshold、默认A_off、历史FAILED不改。未启动120、独立TEST、额外Long训练或正式性能。

导航：[HW实际检查点](../.pipeline/experiments/libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)、[范围核查](../.pipeline/experiments/libero-fullscale-20261017/SCOPE-REVIEW.md)、[文献派生报告](../投机采样/results/runtime/p0-libero-eval-scope-literature-cpu1-v1-20261017-r1/RESULT.md)。原始机器人tensor/相机/模型与完整旧清理收据未作为本次公开物料，不能声称仅凭GitHub已独立重算。
