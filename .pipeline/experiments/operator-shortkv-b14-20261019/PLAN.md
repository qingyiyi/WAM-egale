# 短KV与B14性能核查

1. **已完成首轮：CPU证据定位。** 逐点复核旧/当前全部计时，绑定形状、源码、引擎和计时范围；沿固定Torch 2.9 SDPA/Flash及当前完整block调用链核对布局、split、拷贝和cache。
2. **有限执行/PARTIAL：真实输入诊断。** 六个既有block0 fixture；优先短KV/B14及对应B2，原Torch/当前CUDA核心和完整block同边界交错计时、A/A对照，profile与计时分开。仅根据观察触发最小修复，不做配置大搜索。
3. **已收口：核查与交付。** 涉及改动时逐位/有效KV/metadata/输入/RNG验证，覆盖短中长B2/B14；需要时有限30层回归。保存所有样本、失败、源码差异与资源终态，更新HW及main镜像。

新预算见REGISTRATION.json：单GPU生命周期≤1800秒，绝对墙钟≤4小时，CPU≤2，新增≤512MiB；不延续M26余额。当前lhw占卡，未启动GPU。只修改optimized及共享tools，baseline/投机/原tile只读；四项false不变。
