# 10/20 已执行后继：固定本文件所在commit审查

用户已批准先数值/Target、后原语/迁移与tile交接的两卡有限计划。以下是实际执行结果，不是旧只读提案。

1. [最终单层](../.pipeline/experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：九底层真实签名×两路线有限逐位通过；每路线18合法有效KV转移直接对原Torch通过。默认仍慢native2.68–5.02倍，ready另慢1.2%–6.0%。当前[CUDA源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)/[FFN行组调度](../算子研究/wam_ops/optimized/csrc/ffn_pipeline.cuh)/[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。
2. [数值归因](../.pipeline/experiments/single-layer-async-gemm-20261020/NUMERICAL-ALIGNMENT-RESULT.md)、[原语](../.pipeline/experiments/single-layer-async-gemm-20261020/PRIMITIVE-EFFICIENCY-RESULT.md)、[ready负结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ROW-READY-RESULT.md)、[主张矩阵](../.pipeline/experiments/single-layer-async-gemm-20261020/FOLLOWUP-CLAIM-EVIDENCE.md)、[资源/失败](../.pipeline/experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)。完整3状态/cache强baseline矩阵未完成，不能声称H3/H4或E2E。
3. [RoboTwin原渲染恢复](../.pipeline/experiments/robotwin-pilot-20261020/DRIVER-INTEROP-REPAIR-RESULT.md)：同A100、同原场景，配套535库两次完整专家/相机成功，595 compat对照重现OIDN非法访存。只修复客户端库路径，系统驱动不动；0模型policy/teacher/训练/DEV，Target首6局待执行。旧两次失败保留。
4. [MegaMoE差距审查](../.pipeline/experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：已有常驻/warp分工/双缓冲/行组ready，仍缺K片段ready及中间槽流水。B14专用化存在32KiB缓冲机会，未改算子或增加性能样本；原始单状态action保留原算子。
5. 原LIBERO6000局及其质量缺口不变；四权限仍false。旧FAILED/retention/paused72/MLP0/22缺收据及旧M39资源FAILED不修改。review/RESULTS.json仍是旧固定paused72身份，不拿它代替当前状态。

本次M39完成1380/1500block后停止，43owned根进程核验退出；RoboTwin旧495.321秒失败保留，新增修复对照119.345秒，累计614.666秒；修复后最终两卡无compute进程，无自动后继。源/HW/小JSON/CSV可公开检查；模型/PT/KV/相机/DSO/工具链/私有供应商SASS/连接配置不公开。网页端不能凭汇总表独立重算未公开原始tensor。

HW入口：[短状态](../.pipeline/PROGRESS.compact.md)/[当前Cycle](../.pipeline/cycles/wam-research-20261016/PROGRESS.md)。只把得到源证据支持的事实纳入规划；不要求改阈值、删失败、扩大训练或把负结果写PASS。10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00老师中文证据包。
