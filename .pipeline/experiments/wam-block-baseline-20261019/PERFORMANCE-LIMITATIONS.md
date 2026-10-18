# 性能：本轮没有新GPU时间

PERFORMANCE-RESULT.csv的54行都是NOT_RUN_NO_GPU_AUTHORIZATION，数值为空，不是0ms。benchmark_block/gpu_matrix/gpu_job提供已绑定的预算驱动、A/A、ABBA/BAAB与准备矩阵；本轮只做CPU控制/负例测试，完整GPU执行尚未发生，必须先有新授权、资源准入与fixture/state证书，不能声称已经正式benchmark。

已有R1的54点完整block配对比中位0.984655、几何均值0.969002，范围0.410135–1.343666，9点>1.05；六30层中两例>1.05。只是旧有限测量，不是B0结果。

重新读取固定CSV发现，54个原Torch block每点三轮“轮内中位数”的max/min有45点>1.05、36点>1.10，中位1.172230，最大7.387167。没有独立A/A控制；三轮只有三个执行块，不把每个Graph replay/元素当独立样本。此观察不证明是CPU、GPU、时钟或某个算子根因；不能据此删除旧慢点或挑最快轮。

新方案：先A/A，若三轮配对噪声不能分辨5%门则PERF_PENDING；A/A建议偏差上限2.5%只是新工程准入。再同cache、同eager边界交错R0/R1/B0/合格R_strong，至少三轮，保存全部原始Event和同步wall。prepared execution与含状态恢复/布局/缓存准备的integration时间分开；冷构建另记；正常计时无hook/profiler/逐算子同步。事件区间会包含host launch gap，不冒充纯kernel活跃时间。

完整block Graph原路径存在动态nonzero/tensor-to-Python缓存分支，当前NOT_IMPLEMENTED，不强行capture/跳过CPU更新；已有attention Graph-core与block eager不能相除归因。guarded/shared/static投影组合先逐位/状态验证，不能按名字拼强baseline；既有OriginalShapePack要求真实30层，新的block-only静态诊断不伪造30层对象。

未来B0工程筛查：六核心fixture配对比几何均值≤1.05，无可复现分层>1.10；噪声大/资源异常则UNKNOWN而非PASS。不是旧质量门或统计非劣标准。block层没有g/round提交，不报告ΣT_round/Σg，更不能给机器人E2E加速。

计时依据：[PyTorch2.9 CUDA语义](https://docs.pytorch.org/docs/2.9/notes/cuda.html)和[CUDA12.4.1 Best Practices](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html)。原报告与CSV见SOURCE-MANIFEST，旧失败不改。
