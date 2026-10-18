# 当前算子审查包

本次代码/证据快照：2026-10-18晚。旧review/RESULTS.json保留历史paused72身份，不覆盖。

- M21已经完成54性能点、414数值比较、六B14 query×四引擎完整30层、九保存输入I/O。旧standalone完整30层三例容差失败，全部保留；shared/guarded六例有限逐位/状态通过。详见COMBINED-RESULT。
- M22新用户授权：主要负责真实Torch算子对齐，允许固定上游源码复用；独立v2仅移除fast_math，CPU编译及GPU核验完成：108attention/54block、六完整30层及三短中长完整policy配对全部逐位/有效状态通过；CPU27保存输出复核通过。[新结果](../.pipeline/experiments/operator-torch-alignment-20261018/RESULT.md)。这只证明有限真实输入，不是机器人质量/任意输入证书。新二进制54点和六完整30层已在用户确认空闲窗口实测：attention约1.002；block9/54及full30两例慢>5%，所有慢点、单进程少样本和空载计数异常保留。[性能结果](../.pipeline/experiments/operator-torch-alignment-20261018/performance/RESULT.md)。15项保存输入/原始输出CPU同源核对全等，不声称离线逐张量重算未保存的全部54点。
- 本次不修改原tile源码、全局默认或质量四false。模型权重/原始tensor/二进制/凭据不公开；原结果内绝对路径为服务器证据指针。
- 请按本提交固定版本阅读[网页审查Prompt](OPERATOR_REVIEW_PROMPT-20261018.md)。
