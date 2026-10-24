# 当前讨论（2026-10-23T19:02:24.222118+08:00）

5090隔离环境已安装：Torch2.9.0+cu130、CUDA13.0.88、NCU2025.3.1；Torch/nvcc支持sm120，依赖与构建文件核对通过，GPU kernel0。累计安装22.50分钟，约6.05GiB，owned安装进程退出、日志/锁定清单已回本机。利用率100%原因UNKNOWN，但不阻塞适配；计数器需宿主管理员，未发生重启。两次下载调整FAILED保留，旧研究预算/BAAI队列/四false不变。 [安装结果](../../maintenance/rtx5090-environment-20261023/SETUP-RESULT.md)。用户明确允许利用率异常在不影响工作时暂不处理，当前不要求重启；此前只读/未安装描述为历史状态。

5090环境只读核查完成：原版32GB/170SM，25 CPU额度与90GiB内存，数据盘50GiB可用。现有CUDA12.4/Torch2.5.1未配置SM120；counter受限（无SYS_ADMIN/PERFMON），util100%与0显存/空compute列表不一致，原因UNKNOWN，未启动kernel或NCU。用户条件选为后续性能基准并要求源码/结果回本机，已Maintain；尚未迁移/安装/测量。两槽候选97.75KiB仅容量可容纳，实际适配未验；旧预算、BAAI队列及四false不改。 [结果](../../maintenance/rtx5090-environment-20261023/RESULT.md)。下方未选新卡为此前讨论背景，以本条最新用户选择为准。

用户新批准FFN1尾块移除第三输入槽，已Maintain。当前worktree以L13为基础实际两级编译通过，Shared97.75 KiB；原冻结L13保留，GPU精度/性能未测。Config3只用于保持cp.async类型，实际Mma2，首次模板特化失败保留。[结果](../../experiments/single-layer-async-gemm-20261020/ffn1-two-stage-20261023/RESULT.md)。

10/23硬件讨论补充：用户指出H系列tile/warp实现不能直接搬到4090/5090D，尤其shared资源不同，并询问两种RTX如何选择。已Maintain部署可迁移性反馈；原“先H800研究、再RTX验证”的建议需要明确架构专用后端重写成本，不能视为一次补测。调度思路可以复用，tile/流水/指令及数值与性能需逐架构适配；新卡尚未选择或接入，无新增GPU作业。助手条件建议为同租金/独占/计数器权限下优先原版5090D 32GB；这不写为用户决定。相关官方资料：[Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)、[Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)、[TMA](https://nvidia.github.io/cccl/unstable/cccl/tma.html)。

用户提出guarded不应单独使用CUDA Graph。核对timing.py第56–76行后确认本轮A/L1/L13/D/G均被捕获并重放，包含自有单kernel；内部实现不依赖Graph不等于测试未使用Graph。已解释双方应同条件，普通双方eager对照尚未测；未将疑问写成禁止全部Graph的长期决定，未启动新增测量。[证据](../../experiments/local-warp-pipeline-20261023/MEASUREMENT-MODES.md)。

本轮附件T0–T4已实际完成，FFN2局部warp重叠没有可信净收益，止于独立负候选；副线L13同模板融合两处LN调制小幅正向，只在隔离研究采用。不能将原Torch dense比值代替强G结论，中长仍慢3.4%–3.6%。后续最多讨论primitive差距、FFN2是否止损、可选H800迁移，未启动新方向。完整阶段表按每项/失败输出且版本化保存。

用户询问发表与硬件：原LingBot-VA正式RSS2026、RTX5880Ada部署已由正式PDF核验；Flash-WAM推理L40S训练H100，WAMJET的LingBot-VA行H100，Fast-WAM/AHA是独立相关架构。用户当前没有H100，已Maintain；H800只是候选，不能假定新资源已授权。SM120每CTA99KiB不足当前109.75KiB，Hopper227KiB更便于迁移，但源码SM80/108显式guard仍须适配与重验。

用户问强guarded：已按prepared.py/candidate_attention.cu解释为Torch其余算子＋共享历史G1/G2 attention；G2仅在不多增加K64尾块时使用，私有KV依然隔离。与整层persistent L13不同，非PyTorch官方命名，非宣称全世界最优。

RoboTwin继续原队列，微波炉100组三臂完整后才查原因；16:33只读125完整组，16:32微波炉20组，未启动新闭环任务或更改配方。原160小时/10天及四false保留。

[自包含回传](../../experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[全表](../../experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[先前讨论](../../experiments/local-warp-pipeline-20261023/PRIOR-DISCUSSION-SUMMARY.md)。
