# 当前精度版本的融合阶段计时

- [x] P0：核对真实源码、三KV fixture、旧累计预算与5090空闲。
- [x] P1：仅重建插桩副本及guarded SM120比较入口，不改正常persistent源码。
- [x] P2：真实完整block数值/有效状态核验，三方eager两独立进程交错计时。
- [x] P3：融合阶段globaltimer诊断与原Torch/guarded eager同工作量对照；测插桩扰动。
- [x] P4：融合阶段表、瓶颈排序、固定源码与结果回本机、HW与owned清理。

90分钟含准备/失败/等待/清理；旧累计24832.512/36000秒不重置；CPU4、新存储1GiB、完整block调用至多14000。无CUDA Graph、NCU、SHA、smoke、训练、DAG扩展。

正常整层结果与阶段诊断分列。自有阶段含原同步；Torch阶段eager操作独立测量，不能将比例视为纯kernel比或把各阶段相加为完整层。FFN存在重叠，主表合并整个FFN；Cross KV冷构建单列。guarded若未通过同卡数值/状态，只报告阻塞或诊断，不修改标准。
