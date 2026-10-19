# 当前短入口

更新：2026-10-19T19:27:26.713126+08:00。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先约束记忆，再读当前四文；旧大索引/原trace按需。

## 当前：M31独立optimized与静态tile子图

[实现与结果](experiments/transformer-tile-subgraph-20261019/RESULT.md)、[源码说明](../算子研究/wam_ops/optimized/SUBGRAPHS.md)。baseline及依赖88项冻结不变；optimized独立Python/C++/DSO，默认sequential等价，tiled和FFN1融合独立开关。self/cross/FFN实际子图，同stream阶段执行；attention显式输出tile、FFN1显式MMA tile已实现，QKV/Wo/FFN2仍cuBLASLt，persistent/ready未实现。

9真实block/72KV状态/输入RNG参数/API门有限逐位通过；独立进程9输入/54状态、不加载baseline/旧DSO。首次kernel名识别误判与video标签/收据错误保留，最终新attempt补齐。3GPU作业81.007秒/483block，owned退出。新full30显存不足未测，无新性能/闭环证据；M28旧full30不能继承。M30原DAG/MegaMoE设计仍作来源。

## 已完成：M28完整native block

[完整结果](experiments/block-native-baseline-20261019/RESULT.md)；[源码讲解](experiments/block-native-baseline-20261019/CODE-GUIDE.md)；[九点双进程表](experiments/block-native-baseline-20261019/PERFORMANCE-TABLE.md)。两份活跃入口在 算子研究/wam_ops/baseline 与 optimized，均执行完整C++/CUDA block；Torch保留Tensor/allocator/stream/handle，GEMM直接cuBLASLt，固定上游Norm/Flash/许可证公开。

最终六action各23关键量/六原语、24action/video状态门、12optimized action写读门、六actual30 action query有限逐位过。原模型839个requires_grad=True标志、所有有效KV/metadata/input/RNG/参数地址/安装还原通过。B2 fullquery是verifier root CFG0/7派生；Q128仅block0，不声称新native视频30层/完整policy或闭环证书。

两独立进程各9点、504原始样本齐：动态block相对原Torch两个A/A中位均不慢；A/A最大8.63%偏差保留，不把optimized小差值当稳定新增收益。不与最强shared-prefix/缓存projection/Graph混比；当前动态native有host长度同步，Graph不支持。23GPU作业1088.541/3600生命周期秒，288模型/5114独立block/144原语，全部owned退出。原标量pack/slot的B14回退经profile后用16字节复制/CUB稳定scan修正，这是工程基础，不是已证实MegaTransformer创新。

旧M21 full30、M25严格2.5%/NVML、M26 CUTLASS FFN2与NCU偏差、M27资源PARTIAL以及本轮缺字段/混合dtype拒绝全部保留；M25由新M28范围接替，不修改旧结果。1875投机源和保护文件最终复核不变。

## 远端既定评估

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500完整；[成功率表](experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。Object 1340/1500，完成数T500/C420/P420，来源2026-10-19T11:26:39Z；controller身份匹配=True，原同卡队列继续。GPU1已自动进入前200组C/P后继，仍与matched Target同卡；GPU0既定分片已完成。[动态核对与ETA](experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md)。原CPU约5分钟只读巡检继续，SSH失败不冒充实时。

## 文献与下一步

[M29原文/主张门](experiments/speculative-denoising-novelty-20261019/RESULT.md)：ICML2025投机扩散、ICLR2026 FlowCast、TS-DP/FLASH已有Transformer draft/蒸馏/机器人batch verify/PF等直接先例。新近Speculative Forcing用于视频世界生成，WAMJET已优化LingBot-VA。当前复现/普通batching/缓存/FFN融合不足以构成系统创新。

默认下一步M32尚未启动：资源允许时补新full30门，对齐最强B14 shared-prefix/O3/缓存Graph，再有界实现persistent worker/ready；记录sum(T_round)/sum(g)、完整chunk与质量。新机制仍是假设，不自动启动新训练、架构搜索或大型fusion。当前源码/HW只发布main进度镜像，历史固定科学commit保留身份。

## 科学和资源边界

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧NO_GO/FAILED/retention、paused72/96、MLP0、22缺收据/default A_off不改；TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32冻结。

PF2=SST完整policy，T50更新/51action调用含末尾，非bonus或状态恢复；reset/cache不计周期。所有实验同用512-token BF16等价reset，cold另列；actual796/800罚分分列。首两S严格配对失败、训练/DEV曝光和长尾保留，不将描述性SR当独立TEST或非劣。

投机源码/配置/运行依赖与原tile只读；用户此前允许M28改造baseline且已归档旧源；当前从该完成版继续optimized。state/rules保护、不建缺失cycle.yaml；不升级/旧sync覆盖HW。BAAI写入仅~/cxz；AutoDL不再SSH，实例关机UNKNOWN。不杀他人/改服务/GPU时钟；本机计时自然空闲、精度可共享。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB，最多每卡2worker；M28已终态，不重置原截止。无自动120/独立TEST/新TF正式45chunk/第二环境。中文论文仍会话交飞书，不新建论文初稿。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。

约束记忆入口：[memory/INDEX.md](memory/INDEX.md)；完整结果和旧历史按Cycle索引读取。
