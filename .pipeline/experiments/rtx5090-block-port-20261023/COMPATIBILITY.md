# SM120迁移修改

- 当前候选：L13＋FFN1尾块两级缓冲。原A100二进制不搬到SM120直接使用。
- 构建架构改为12.0；CUDA13.0.88/Torch2.9.0+cu130。设备数学源码未改。
- 移植副本CUTLASS host adapter的版本条件修复：CUDA>=12.5应包含13.0；原条件错误检查minor>=5。只影响宿主驱动包装器选择。
- 项目环境补lib/libnvvm.so到libnvvm.so.4，以及nvvm/lib64到../lib；实际nvlink载入路径诊断支持此布局修正。没有修改驱动/系统库。
- 本轮codegen移除隐式SHA；保留原文件/源码来源，只记录用户要求未计算。增量重链可显式WAM_REUSE_GENERATED=1，不重生成已冻结头文件。
- 从原model.py AST执行原custom_sdpa/FlexAttnFunc/WanAttention/WanTransformerBlock类体；只读取真实block0参数，不加载完整模型或修改投机侧。
- 首轮host兼容失败、两轮NVVM库布局失败、插桩版首次链接失败及一次诊断命令前缀解析错误保留。
