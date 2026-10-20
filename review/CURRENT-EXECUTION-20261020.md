# 10/20 已执行后继：固定本文件所在commit审查

用户已批准先数值/Target、后原语/迁移与tile交接的两卡有限计划。以下是实际执行结果，不是旧只读提案。

1. [最终单层](../.pipeline/experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：九底层真实签名×两路线有限逐位通过；每路线18合法有效KV转移直接对原Torch通过。默认仍慢native2.68–5.02倍，ready另慢1.2%–6.0%。当前[CUDA源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)/[FFN行组调度](../算子研究/wam_ops/optimized/csrc/ffn_pipeline.cuh)/[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。
2. [数值归因](../.pipeline/experiments/single-layer-async-gemm-20261020/NUMERICAL-ALIGNMENT-RESULT.md)、[原语](../.pipeline/experiments/single-layer-async-gemm-20261020/PRIMITIVE-EFFICIENCY-RESULT.md)、[ready负结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ROW-READY-RESULT.md)、[主张矩阵](../.pipeline/experiments/single-layer-async-gemm-20261020/FOLLOWUP-CLAIM-EVIDENCE.md)、[资源/失败](../.pipeline/experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)。完整3状态/cache强baseline矩阵未完成，不能声称H3/H4或E2E。
3. [RoboTwin迁移接续](../.pipeline/experiments/robotwin-pilot-20261020/RESUME-FIRST6-PLAN.md)：16:24北京固定专家初态114/114齐，首个Target case完整成功，8policy/240命令、184.197秒；初始物理/观察逐位通过，GPU1第二局stack_blocks_two正在运行。原渲染修复/旧FAILED保留；[专家None抓取异常](../.pipeline/experiments/robotwin-pilot-20261020/EXPERT-SELECTION-CONTINUATION.md)按官方筛选规则从下一seed继续，不重跑旧seed或改变截止。首case旧视频导出入口在0teacher/0动作处PARTIAL，用户已批准[修正后补跑](../.pipeline/experiments/robotwin-pilot-20261020/FIRST-CASE-RETRY.md)，旧原件与截止保留。108条完整512 bank已构建并完成张量拷贝核对，完整任务缓存等价未实测。后继TF训练/DEV未完整实现。
4. [MegaMoE差距审查](../.pipeline/experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：已有常驻/warp分工/双缓冲/行组ready，仍缺K片段ready及中间槽流水。B14专用化存在32KiB缓冲机会，未改算子或增加性能样本；原始单状态action保留原算子。 [填尾波设计](../.pipeline/experiments/single-layer-async-gemm-20261020/TAIL-FILL-SCHEDULING-REVIEW.md)仍待验证：空闲CTA取ready任务，self关键路径优先，独立cross K/V后备；缓存公平，未启动新算子GPU。
5. 原LIBERO6000局及其质量缺口不变；四权限仍false。旧FAILED/retention/paused72/MLP0/22缺收据及旧M39资源FAILED不修改。review/RESULTS.json仍是旧固定paused72身份，不拿它代替当前状态。

M39仍1380/1500block终态。RoboTwin已闭合GPU账本2266.544秒，运行中另计，原24h/准备18:20北京不重置；首六局沿原17:46:42北京deadline，预留完整episode与清理。当前自动链只到原Target首六局，不能声称174局均已排队。源码/HW/小JSON可公开，模型/PT/KV/图像/DSO/工具链/私有SASS/连接配置不公开。

HW入口：[短状态](../.pipeline/PROGRESS.compact.md)/[当前Cycle](../.pipeline/cycles/wam-research-20261016/PROGRESS.md)。只把得到源证据支持的事实纳入规划；不要求改阈值、删失败、扩大训练或把负结果写PASS。10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00老师中文证据包。
