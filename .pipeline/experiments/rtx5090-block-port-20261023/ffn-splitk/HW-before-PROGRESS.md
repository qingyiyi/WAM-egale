---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T23:45:25.867780+08:00'
plan: PLAN.md
current: M39-FFN-TMA-PIPE
next: 保留GRAIN；已核Torch的480/672 CTA与K分段，建议先局部split-K执行组织、再隔离TMA寄存器流水，未启动GPU；双A100原队列继续。
---
# 当前进度

10/24 00:04只读续项：确认新TMA缺少原CUTLASS显式下一fragment预取、三级变两级并增加交接；三个小投影实际八MMA warp，FFN2四warp。原Torch trace另证实输出投影480CTA/K十段、FFN2 672CTA/K七段；自有96输出tile在CTA内串行K段。建议优先局部split-K计算并行且按序BF16提交，再隔离TMA主循环；各因素收益尚未定量，未实现/运行。已固定官方SM120 BF16 TMA参考；详见[诊断](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/TMA-GEMM-DIAGNOSIS.md)，原终态/完整表/预算不变。

本轮三个FFN流水候选已完成并收口，均有限逐位/状态通过但完整层退化：INPUT约2.7%–3.2%、EPILOGUE约3.5%–4.0%、KREADY约3.5%–3.8%。当前保留GRAIN-M64-FFN2，源码与public入口已恢复；单输出槽与两输入槽放得下，未缩tile、未做尺寸网格。

最后同场原Torch/GRAIN短中长1.0892/1.5417/1.9568倍，仍慢guarded约4.03%/5.56%/7.45%。三方eager无Graph、冷CrossKV单列；候选阶段已测，当前GRAIN逐阶段为22:25历史值并明确标注，不把跨轮细微差异当因果。完整17行表逐候选已在聊天打印并归档。

三真实B14/Q16 block0的26中间量＋输出、KV/输入/RNG及两public流有限通过；KREADY14片段计数正确。一次诊断生成器编译失败保留，修复只改诊断字段；三个正常源码无数值失败。四科学权限与production_admitted=false。

自有进程自然退出、5090显存0，源码/二进制/结果回本机6份byte一致。完整block 22113/30000次；保守本轮37分钟，累计29686.413/36000秒，余额1.754小时。旧deadline不重置，不自动追加作业。

双A100 RoboTwin仍按原队列接续，23:47只读见两卡模型/模拟上下文、CPU巡检正常；本轮不重算SR、不改队列。

下一步：讨论同M64下原CUTLASS与新流水的等待/寄存器预取/提交差异，再决定是否单因素M32；容量不是本轮已证实瓶颈，不启动大DAG或无边界搜索。

[结果](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/SELECTED.json)。
