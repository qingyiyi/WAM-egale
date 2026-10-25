# FFN 两阶段同步版

- [x] 读取现行约束，聚焦 WAM；确认5090无其他CUDA进程。
- [x] 保存独立原A源码/二进制，修改optimized阶段边界。
- [x] 编译S与独立18阶段观察器；精度先行。
- [x] 三KV、两独立进程，Torch/Guarded/A/S交错eager完整层计时。
- [x] 分开FFN1+GELU与FFN2+残差的阶段表、归因及采用决定。
- [x] 回传核验、owned清理、HW短更新。

本轮仅增加FFN1完成后的grid同步。保持global ACT、FFN1主体8/尾4MMA、FFN2四MMA、七段K2048有序BF16递推；不同时做DSM、统一warp角色或重写GEMM。FFN1无残差，FFN2最后一段处理bias和gate/残差。

90分钟生命周期上限（含准备失败/清理），4CPU、7000完整block调用、2GiB新增。完整层由无插桩二进制测量；阶段是独立观察器与Torch独立eager片段，不把阶段比值当完整层净收益。采用准则：精度通过优先，若三KV相对A中位延迟回退均≤3%，可采用易分析阶段版；若超出则保留为诊断版并回退A，原始结果保留。此3%为本轮工程筛选，非论文统计标准。

已完成并采用S；实测与边界见REVIEW_RETURN.md、STAGE-TABLE.md，原A保留。
