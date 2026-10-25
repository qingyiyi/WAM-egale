# FFN1 专职 warp 流水对照

- [x] 聚焦 WAM Cycle，确认当前保留 S、独立二进制及三真实输入；读取旧 FFN2 流水负结果。
- [x] 实现 W：仅 FFN1 的输入和结果交接，1T/4M/3E。
- [x] 编译并先核27项/七段/状态/角色计数；失败即停止计时。
- [x] 三KV、两独立进程无Graph完整层与分阶段测量。
- [x] 采用或回退；完整阶段表、原始数据、本机回传、HW/GitHub及owned清理。

## 设计与判定

当前 S 是对照，非旧A。W使用448个64×128输出tile、完整K3072，每tile48个K64；相对S的120个混合大tile，比例和tile同时改变，因此不是单因素warp数量因果实验。选择64×128是复用已核验的四warp Tensor Core fragment与82KiB双输入/单结果槽，避免128×256完整结果超shared容量，不做网格搜索。

W0装载下一K；W4–7完整归约当前输出tile；W1–3对上一输出tile执行原bias、BF16舍入、GELU、global写回。一个结果槽足以允许下一tile的寄存器MMA与上一tile epilogue重叠；覆盖结果槽前等EMPTY。两输入槽独立READY/EMPTY，不与结果union重叠。仍先完成全FFN1再grid同步，后做原FFN2；不新增DSM或完整DAG。

FFN2计算不改，ACT完成计数按新真实tile数修改为四组112；这是新审计schema，不伪装原28/28/28/36。每CTA debug记录实际角色循环计数。性能含新增tensor map host准备；profile与正常二进制分列。沿JOB.json的100分钟生命周期/7000调用/2GiB/4CPU上限，失败和等待计账。

旧INPUT/EPILOGUE/KREADY、B-INPUT/C-OVERLAP均是FFN2 K段TMA/handoff实验，已负结果；本次作用于FFN1完整K输出tile，不重复其名称或结论。
