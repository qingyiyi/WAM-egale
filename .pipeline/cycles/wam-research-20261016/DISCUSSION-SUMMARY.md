# 当前讨论（2026-10-23T16:36:22.689192+08:00）

用户提出guarded不应单独使用CUDA Graph。核对timing.py第56–76行后确认本轮A/L1/L13/D/G均被捕获并重放，包含自有单kernel；内部实现不依赖Graph不等于测试未使用Graph。已解释双方应同条件，普通双方eager对照尚未测；未将疑问写成禁止全部Graph的长期决定，未启动新增测量。[证据](../../experiments/local-warp-pipeline-20261023/MEASUREMENT-MODES.md)。

本轮附件T0–T4已实际完成，FFN2局部warp重叠没有可信净收益，止于独立负候选；副线L13同模板融合两处LN调制小幅正向，只在隔离研究采用。不能将原Torch dense比值代替强G结论，中长仍慢3.4%–3.6%。后续最多讨论primitive差距、FFN2是否止损、可选H800迁移，未启动新方向。完整阶段表按每项/失败输出且版本化保存。

用户询问发表与硬件：原LingBot-VA正式RSS2026、RTX5880Ada部署已由正式PDF核验；Flash-WAM推理L40S训练H100，WAMJET的LingBot-VA行H100，Fast-WAM/AHA是独立相关架构。用户当前没有H100，已Maintain；H800只是候选，不能假定新资源已授权。SM120每CTA99KiB不足当前109.75KiB，Hopper227KiB更便于迁移，但源码SM80/108显式guard仍须适配与重验。

用户问强guarded：已按prepared.py/candidate_attention.cu解释为Torch其余算子＋共享历史G1/G2 attention；G2仅在不多增加K64尾块时使用，私有KV依然隔离。与整层persistent L13不同，非PyTorch官方命名，非宣称全世界最优。

RoboTwin继续原队列，微波炉100组三臂完整后才查原因；16:33只读125完整组，16:32微波炉20组，未启动新闭环任务或更改配方。原160小时/10天及四false保留。

[自包含回传](../../experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[全表](../../experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[先前讨论](../../experiments/local-warp-pipeline-20261023/PRIOR-DISCUSSION-SUMMARY.md)。
