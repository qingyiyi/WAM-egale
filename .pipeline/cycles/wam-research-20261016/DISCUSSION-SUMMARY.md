# 当前讨论与授权

用户最新要求先尝试对齐逐元素和FFN2，再结合远端结果汇报；先对齐再调度，失败必须给可追溯原因。
真实Torch不是native代理。独立体能对齐不能推出只差同步：固定CTA数、shared/寄存器、编译与cache都须区分。
仅优化简易单层optimized，baseline/原tile/LIBERO不改。

本轮已完成：vec4有完整层收益，FFN2独立接近Torch但整层无稳定收益，后者撤回。非数值失败，也未证明单kernel理论上不能对齐。
RoboTwin新60分钟窗口只覆盖valid08，已完整终态并清理；原24h总预算不增。parent18/CHILD/CONT/固定接口CAL齐，
后继CURRENT18需原09:30截止之外的新窗口，不能借单局许可延长全部链。所有自然失败和旧记录保留。

M39累计10h、M40累计24h约束仍在；余量不是无边界搜索授权。四项false、retention、原FAILED、默认A_off不变。
HW Maintain已保存用户优先级和单局窗口；不把本轮诊断假设当长期科学结论。GitHub仅main核对源与小证据，不上传权重/凭据。
