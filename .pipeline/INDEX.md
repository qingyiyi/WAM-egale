# WAM 投机与单层 Transformer 当前入口

更新：2026-10-20T17:26:40+08:00。

先读[上下文策略](CONTEXT-POLICY.md)、[记忆索引](memory/INDEX.md)及当前约束，再聚焦[Cycle进度](cycles/wam-research-20261016/PROGRESS.md)、PLAN/EXECUTION/DISCUSSION。旧长索引/原始trace按需读，v15不升级或全量sync。

- [当前CUTLASS设备GEMM结果](experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)：整层一个kernel，9真实签名/18有效KV有限逐位通过。B14新整层7.94/8.14/8.64ms，同轮native2.65/2.94/3.39ms；仍慢2.54–3.02倍。新ready未验、默认关闭，非强cached/Graph或E2E。
- [当前前版profile](experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/PROFILE-RESULT.md)：GEMM70.70%–76.08%，cross K/V50%–54%。只作插桩阶段归因。用户明确复用成熟设备mainloop、保持单kernel，见[决定](memory/global/requirements/decision-继续算子优化并先核对当前瓶颈.md)。
- [资源闭合](experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/RESOURCE-AND-STOP.md)：累计1497/1500block，首CPU构建FAILED保留，本次owned根进程全回收，两GPU空。下一GPU阶段须有有界预算；原8h/24h和deadline不重置。
- [RoboTwin首六局](experiments/robotwin-pilot-20261020/FIRST6-TARGET-RESULT.md)：原Target 5成功1完整失败，83policy/6557teacher；实际命令/罚分分列。114专家初态和108完整bank齐，完整任务cache等价未测。TF适配/训练/DEV未完成，不在排队。
- [四套LIBERO原质量](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，T/C/P各500。科学审计未放行，不是独立TEST。

下一步：先新GEMM阶段/epilogue/local-memory瓶颈和B14专用化，再跨阶段ready填尾波。普通缓存不是新系统贡献；完整1/3/7×cache强baseline矩阵缺。仅单层，不自动30层或机器人集成。当前[source guide](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。

四项quality/test/noninferiority/performance均false，旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保留。原始action沿原算子。baseline/依赖、投机代码、旧tile/HW和state/cycle/rules不动。BAAI写入限~/cxz，AutoDL已归还，不改驱动/系统/时钟、不杀他人。

只[GitHub main](https://github.com/qingyiyi/WAM-egale/tree/main)镜像核对源码/HW/小证据，不发布模型/PT/DSO/凭据。旧[最终版](experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)与[本次前历史](cycles/wam-research-20261016/history/device-gemm-followup-start-20261020/PROGRESS.md)可追溯。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00老师中文证据包；[日期/作者缺项](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。不保证投稿/录用。
