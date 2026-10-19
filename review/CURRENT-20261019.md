# 10月19日当前进度导航：独立optimized与静态tile子图

最新M31：[实现/验证结果](../.pipeline/experiments/transformer-tile-subgraph-20261019/RESULT.md)、[新源码与使用](../算子研究/wam_ops/optimized/SUBGRAPHS.md)。baseline及依赖88项冻结；optimized不再继承baseline，独立DSO，seq/tiled/FFN1分开。9真实block/72KV状态及输入RNG/API门逐位通过，独立进程不导入baseline/旧WAM库；attention和FFN1显式设备tile已实际运行。QKV/Wo/FFN2仍cuBLASLt，persistent未实现。新full30因约4.9GiB显存不足未测；没有新性能/闭环证据，不继承下文M28 full30。首工具误判/标签问题及原attempt保留。3 GPU作业81.007秒/483block，owned全退出。导出边界见[清单](OPERATOR-M31-EXPORT.json)。

以下M30设计和M28实测为各自固定版本的历史证据，不能替代M31的新门。

最新补充：[M30真实结构/形状/DAG交接](../.pipeline/experiments/transformer-tile-handoff-20261019/HANDOFF.md)、[综合HW的tile/persistent设计](../.pipeline/experiments/transformer-tile-handoff-20261019/DESIGN.md)、[工作大纲](../.pipeline/experiments/transformer-tile-handoff-20261019/WORK-OUTLINE.md)、[MegaMoE官方固定源码映射](../.pipeline/experiments/transformer-tile-handoff-20261019/MEGAMOE-NOTES.md)。只做CPU核验与设计，未实现新persistent kernel；后续用户要求按Transformer自身依赖覆盖self/cross-attention，当前以[attention子图设计修订](../.pipeline/experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)为准：先完整DAG与attention输出tile入口，再相邻投影/归约就绪及常驻调度；FFN2旧数值缺口不阻塞attention研究，baseline留作对照。

已完成三套首轮[成功率](../.pipeline/experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)：每臂500，Long96.6/96.8/94.6%、Goal92.6/92.8/93.4%、Spatial94.4/93.2/92.8%，顺序T/C/P；描述性覆盖不解除原配对/曝光/组件审计缺口，四false不变。Object动态数看当前Cycle，不用下文旧快照代替实时。

main是本地核对后的代码/HW中转镜像。请固定当前文件所在commit审阅；旧ca5f37e原模型参考、4f04 paused72科学快照、旧FAILED/retention及四项false保留身份。

先读[M28完整结果](../.pipeline/experiments/block-native-baseline-20261019/RESULT.md)、[完整路径与源码讲解](../.pipeline/experiments/block-native-baseline-20261019/CODE-GUIDE.md)、[九点双进程表](../.pipeline/experiments/block-native-baseline-20261019/PERFORMANCE-TABLE.md)，再读[两份实现](../算子研究/README.md)。这次baseline/optimized均完整原生block，不再只有attention替换。

六action各23关键量/六原语、24action/video状态门、12optimized状态门、六actual30 query有限逐位通过。最终保留原839个requires_grad=True标志的18模型调用也通过，全部有效KV/metadata/input/RNG/参数地址/安装还原核对。B2完整query是真实verifier root CFG0/7派生；视频只验block0，不冒充新完整policy/视频30层。

最终向量copy/CUB稳定compaction版两独立进程各9点、504样本齐；动态block相对两原Torch中位均不慢。A/A最大8.63%偏差保留，不将小差值当稳定收益。比较包含动态KV/Python开销，尚未与最强shared-prefix/缓存Graph比较，也没有新30层/chunk/episode速度。GEMM直接cuBLASLt，Torch allocator/stream仍是依赖；固定上游Norm/Flash/许可证公开。当前动态长度同步不支持Graph。

M28的23 GPU子作业1088.541秒、288模型/5114独立block/144原语，owned全部退出；旧所有失败/资源中断保留。投机1875源与保护文件未改。用户本次明确允许改算子baseline，替代M26“baseline冻结”的阶段范围；原M26 FFN2失败、M25严格性能缺口不改。

[相关工作核验及主张门](../.pipeline/experiments/speculative-denoising-novelty-20261019/RESULT.md)：ICML2025投机扩散、ICLR2026 FlowCast、TS-DP、FLASH直接覆盖多个核心组成；Sequoia/Cascade已有树硬件成本/共享prefix。新近Speculative Forcing和WAMJET亦需纳入。当前复现和普通融合不能单独作为MegaTransformer创新；候选感知执行超过强B14仍需实验。

[远端核对](../.pipeline/experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md)：Long/Goal/Spatial各1500完整，Object快照1075/1500；以文内时点为准，不把完成数当成功数。后继已自动排队；前200组与matched Target同GPU1，GPU0已完可放置分片。首两S严格配对失败/训练DEV曝光与长尾均保留。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。不新增训练/机器人episode/120/TEST/新TF正式45chunk，不切生产default A_off。下一步先统一最佳B14执行边界，再只选一个profile支持的候选调度机制，主指标sum(T_round)/sum(g)。

[导出范围](OPERATOR-M28-EXPORT.json)：原tensor/权重/二进制/大profiler trace省略，原件在服务器；公开小JSON/CSV/源身份和失败日志，不能据此声称GitHub独立重算。以前M26的[导出清单](OPERATOR-M26-EXPORT.json)及固定提交仍保留。
