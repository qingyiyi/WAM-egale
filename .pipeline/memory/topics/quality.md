# 投机与质量

先读[现行约束](../CURRENT-CONSTRAINTS.md)。以下按任务选用，不自动逐项全读；当前数字读Cycle。

- [TF64研究身份；不预设优于MLP](../global/requirements/decision-TF64固定为主模型与训练收口.md)
- [套件CAL与完整DEV](../global/requirements/decision-按套件有限校准接受阈值并以完整DEV验收.md)
- [Goal多初态与有效样本定义](../global/requirements/decision-本机速度评估采用多初态并汇总有效样本.md)
- [最新本机恢复授权；具体截止看登记](../global/requirements/decision-本机Goal失败先修复再有界接续.md)
- [数据来源、sigma、阈值与PF边界](../global/knowledge/decision-训练数据与接受标准.md)
- [FIT/CAL/DEV/TEST独立性](../global/knowledge/decision-训练验证与LIBERO标准评测的区别.md)
- [少去噪步数对照](../global/requirements/requirement-投机须与少步数Target比较质量成本.md)
- [所有臂共同reset优化](../global/requirements/decision-实验默认启用等价reset优化.md)

PF2是两次完整投机policy chunk后一次canonical target chunk，与bonus不同，也不能恢复已偏离的机器人状态。旧局部成功、CAL接受率或减少teacher调用均不构成最终质量放行。当前不新增训练/TEST/120局。

[四套LIBERO同配方与全量身份](../global/requirements/decision-四套LIBERO同配方TF与三臂评估.md)；[LIBERO范围与历史证据](../global/knowledge/decision-LIBERO评测范围依据与Long证据边界.md)。M41小面板不能覆盖原全量质量身份。
