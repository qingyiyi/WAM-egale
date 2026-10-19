---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-19T17:50:34.115962+08:00'
current: M30
execution_status: complete_CPU_handoff_no_GPU_prototype
next: register_one_bounded_tile_worker_and_FFN2_gate
---
# 当前研究进度

当前计划：[PLAN.md](PLAN.md)。

## M30本次交付

[M30交接](../../experiments/transformer-tile-handoff-20261019/RESULT.md)、[具体设计](../../experiments/transformer-tile-handoff-20261019/DESIGN.md)、[工作大纲](../../experiments/transformer-tile-handoff-20261019/WORK-OUTLINE.md)。核验9份既有逻辑fixture、27参数、21节点DAG，真实D3072/H24/FFN14336。action Q16，B2/B14令Linear M32/224；attention每候选仍Q16。action有效K288/1152/2160；video Q128的本批有效K272/1136/2144，不能混写pool容量。

MegaMoE/DeepGEMM固定1fc9c66975ee322d628b3288534ca0e9ce12a64d，10份源码/许可证只读；观察任务描述、ready、warmup与warp分工。官方BF16路径为SM100，本项目仅借鉴思想、按SM80实现，不移植专有原语。

建议首版显式tile坐标的FFN1+GELU常驻worker；FFN2设备严格门通过后，按一个m_tile全部activation就绪释放FFN2，保留完整K归约，再视证据细化K块/warp分工。具体设计仍为假设。CPU-only，无新kernel、GPU作业、机器人任务或性能数字。首metadata字段工具错误与首次公开镜像保护核对方法错误均保留，最终边界核对1875投机/92算子数学文件与state/rules未变。

## 已完成的系统证据

[M28](../../experiments/block-native-baseline-20261019/RESULT.md)：baseline/optimized均完整native顺序链；Norm/RoPE/pointwise/KV/Flash CUDA、直接cuBLASLt。Torch Tensor/allocator/stream/handle仍依赖；不是单kernel或30层fusion。

六action各23关键量/六原语、24action/video状态、12optimized门、六actual30 query有限逐位。原839个requires_grad=True标志、全部有效KV/metadata/input/RNG/参数地址/安装还原通过；B2完整query为verifier root CFG0/7派生，video仅block0。

两独立进程九点/504原值，动态block相对两原Torch中位不慢；A/A最大8.63%偏差保留，不将小差值当稳定优化。没有与最强B14/cached/Graph统一；当前dynamic native有host长度同步，不支持Graph。23GPU作业1088.541秒、288模型/5114独立block/144原语，owned全部退出。

M21 full30、M25严格2.5%/NVML、M26 device FFN2/NCU偏差、M27资源PARTIAL及M28工具/资源错误均保留。M26 FFN1+GELU约3%是耗时下降，不能说自有版慢3%。[M29直接先例与主张门](../../experiments/speculative-denoising-novelty-20261019/RESULT.md)：普通batching/缓存/FFN融合不能重新称为MegaTransformer创新。

## 质量线

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500完整；对应96.6/96.8/94.6%、92.6/92.8/93.4%、94.4/93.2/92.8%。[冻结汇总](../../experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。首两S严格配对FAILED、训练/DEV曝光、组件审计失败与长尾均保留；描述性覆盖不是独立TEST或非劣。

Object 1166/1500；T500/C333/P333为完成数，非成功数，来源2026-10-19T09:46:38Z；原同卡队列继续。[既有远端核对](../../experiments/speculative-denoising-novelty-20261019/REMOTE-STATUS.md)是较旧时点；动态来源为原CPU只读watch的LATEST，约5分钟更新。后继自动，不通过重复Target或改变同卡约束填卡。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost及BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾，reset/cache不计周期；所有实验同用512-token BF16优化reset，cold另列。actual796/800罚分分列，不用更快失败算加速。

## 下一步与保护

下一步M31尚未启动：先统一强B14边界，再有界tile worker/FFN2门；主要机制指标ΣT_round/Σg，g不能由7候选或环境步代替。只在通过分级正确性/质量后进入对应正式性能。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1。BAAI仅~/cxz，AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲，精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖与原tile只读；不覆盖state/rules、不建缺失cycle.yaml、不升级或旧sync覆盖HW。当前只维护baseline/optimized；M28完整native baseline作为对照，后续优化限定optimized。 不自动训练/120/独立TEST/新TF正式45chunk/第二环境/大融合；中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。原完整PROGRESS与逐项M18–M27见[归档](history/M30-before-handoff-20261019/PROGRESS.md)，旧失败不被派生摘要覆盖。
