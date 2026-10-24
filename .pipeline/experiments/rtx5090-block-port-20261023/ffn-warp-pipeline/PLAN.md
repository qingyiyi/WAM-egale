# FFN局部TMA流水：已授权有界执行

- [x] INPUT：保留M64×N128、四MMA warp、七K2048；独立producer/full-empty，先保留8warp提交。
- [x] EPILOGUE：单独输出槽，两warp后处理与下一段MMA重叠。仅上一项正确后继续；负性能先定位，不强行采用。
- [x] KREADY：2×7个片段计数；固定FFN1生产优先和FFN2七段顺序，不扩整层DAG。
- [x] CLOSE：正常整层三KV双进程、profile诊断分离、完整阶段表、源码回传与owned清理。

冻结GRAIN源码与独立二进制；原Torch/guarded/GRAIN/新候选均eager。每个候选最多一次有依据修正，固定三fixture逐位/状态后才计时。每个完成/停止候选在聊天打印完整17行阶段表，未重测标历史。若INPUT明显退化，先分析是否下一步可针对该原因；无依据则停止后继。新增生命周期≤120分钟、总累计≤10小时、CPU4、持久化≤2GiB、完整block≤30000次，准备/失败/清理计入；以JOB绝对deadline为准。用户允许容量不足时缩小tile；先核活跃缓冲，最多一个有依据的容量修正，不做tile sweep。

当前采用GRAIN，候选未经测量；四科学权限false。冻结原始负结果和deadline，baseline/投机/原tile/保护文件不改。

最终：三候选完成、负性能不采用，恢复GRAIN；一次诊断编译失败保留。具体预算/时间见RESOURCE-AND-STOP-REPORT.json。
