# 中文论文协作的当前证据入口

本页整理于2026-10-25，研究快照截至15:48北京时间。仓库是公开进度镜像，不是实时作业状态。先固定本页所在commit，再按主题读取。后来获得新报告时比较增量，不用旧paused72或旧RESULTS.json覆盖本页时点。

将[PAPER_COLLABORATION_PROMPT.md](PAPER_COLLABORATION_PROMPT.md)复制到网页端：助手先理解项目与证据，首次不写正文；用户指定小节后逐段讨论。中文稿由用户整理到飞书，此次没有新建论文正文。

## 研究主线和证据边界

冻结LingBot-VA/WAM；连续动作候选与验证试图重组小action工作负载，为候选感知的单层Transformer tile/persistent执行创造机会。算法与系统各有测量，完整闭环协同因果链尚未证明。系统范围是一个完整block，不是30层大融合；新单层尚未接入机器人路径。

四项标志均false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。经验局部阈值不是任务成功保证，单层重放不等于机器人E2E。公开材料未含全部KV/velocity或权重，不能据此声称独立复算。

## 投机与质量

- [五环境5090有限DEV](CURRENT-RTX5090-SPEC-SPEED-20261025.md)：44完整三臂组、132主表局，另有PARTIAL/未配齐记录。TF/PF2未获全面质量放行；不是独立TEST。
- [源码导航](../投机采样/docs/CODE-MAP-20261014.md)、[draft结构](../投机采样/experiments/draft_architecture_v1/models.py)、[训练](../投机采样/experiments/tf_effect_train_v1/train.py)、[部署adapter](../投机采样/experiments/tf_effect_train_v1/adapter.py)、[路径选择](../投机采样/experiments/prefix_acceptance_v1/selection.py)。它们用于定位；各实验实际加载版本仍以manifest和导入链为准。
- [CAL资格](../.pipeline/experiments/suite-acceptance-calibration-20261025/CAL-ELIGIBILITY.md)、[阈值规则](../.pipeline/experiments/suite-acceptance-calibration-20261025/CALIBRATION-RULE.md)：五套83030行部署CAL已做有限重放；主池CONT3000生成，不是当前ON3000的on-policy数据。保留FIT隔离、数据暴露与旧权重身份限制，不从query覆盖率推断在线SR。
- [Goal固定多初态面板](../.pipeline/experiments/suite-acceptance-calibration-20261025/multi-init-20261025/prep02/PLAN.md)：T/OLD/Q75/Q90，主目标为成功率、长度、接受深度与失败，耗时附带记录。原10完整局、7初态无效case、1用户中断，仍0合格四臂组。resume03在14:59 VAE路径缺config、0teacher/0新episode准备失败，supervisor收尾误终止自身，旧RUNNING状态不可用。本次已独立归档并清理临时权重，未重试/未延长15:45:32截止。[终态核对](../.pipeline/experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume03/RECONCILE-RESULT.md)。
- [RoboTwin 15:42快照](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/ROBOTWIN-PROGRESS.json)：549完整组三臂/1647局，600组/1800局尚未完成，115.389/160GPU小时，双guard当时alive。原缺口和未配齐记录单列。

## 实际保留的单层系统

保留A：SELF-QKV-TMA-4MMA，5090/SM120、170CTA。原Torch、项目guarded与自有路径同缓存/eager；真实B14、Q16、KV288/1152/2160有限fixture。候选数不是attention序列长度。

- [当前源码](../.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[CUTLASS tile](../.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh)、[公共调用](../.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/whole_layer.py)。不要误用顶层旧optimized作为当前SM120版本。
- [全17阶段表](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/STAGE-TABLE.md)、[最终结果](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/REVIEW_RETURN.md)、[源码和原始标量入口](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/SOURCE-AND-RESULT-ENTRY.json)。服务器绝对路径映射到仓库相对路径；.pt和DSO不公开。
- 本轮A相对原Torch完整单层1.253/1.756/2.186倍；不是TF机器人加速。阶段插桩含屏障，不直接求和当无插桩整层。
- 当前FFN仍保留global ACT；行就绪允许FFN1/GELU与FFN2重叠，不必等全部ACT完成。FFN2保留七段K2048和原BF16舍入契约，具体顺序读源码。

## 负结果和未完成机制

- [DSM8/R1](../.pipeline/experiments/sm120-ffn-next-20261024/DSM-STREAM/REVIEW_RETURN.md)：有限精度通过，R1整层仍慢31%–45%，不采用。
- [CTA单因素](../.pipeline/experiments/sm120-ffn-next-20261024/CTA-ATTRIBUTION-20261025/REVIEW_RETURN.md)：仅170→152CTA使整层慢4.5%–15.5%，168任务attention的额外波次有源码/对照支持；不是所有DSM退化的解释。
- [四CTA/global混合](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/REVIEW_RETURN.md)：B4只改cluster启动；C4局部DSM复用且保留global。有限数值/七段/有效KV/RNG通过，但C4整层比A慢41.12%/48.11%/33.87%，NO_GO。7122调用、28.98分钟，归档与自有进程闭合。逻辑ACT重复读取少75%不等于实测HBM少75%，没有硬件流量计数证据。

可根据实现逐段写定义、流程、方法与可追溯实验。不能写DSM已加速、完整动态DAG已采用、TF必优于MLP或整体无损；算法系统交互、独立质量与最终E2E仍需对应证据。

## HW与增量更新

[短进度](../.pipeline/PROGRESS.compact.md)、[计划](../.pipeline/cycles/wam-research-20261016/PLAN.md)、[完整进度](../.pipeline/cycles/wam-research-20261016/PROGRESS.md)、[讨论](../.pipeline/cycles/wam-research-20261016/DISCUSSION-SUMMARY.md)。旧记录保留身份；新材料先核对版本及其影响的已写段落，再继续写作。


最新[双CTA混合实验](../.pipeline/experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)为NO_GO：M64大tile对照不同于旧M16整ACT驻留E；B2启动近似无成本，C2比A慢12%–15%，G2仍慢约9%–10%。七段数值/27tensor/有效KV/RNG有限逐位通过；17phase/10413调用/105文件回传清理闭合。内部软件计时有观察器扰动，R0/R1均保留；不能把差值直接当DSM指令时间。当前A不改。
