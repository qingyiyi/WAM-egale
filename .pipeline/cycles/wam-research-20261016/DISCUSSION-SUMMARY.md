# 当前讨论（2026-10-22T18:16:14.446638+08:00）

用户批准继续融合FFN1/GELU与FFN2/残差，按四组依赖方案执行。最终R2保留原GEMM与BF16算术，28个FFN1 tile构成组，组就绪后派24个FFN2半段；两段MMA并行，第二段仅提交时等第一段。每CTA先完成自己的生产任务再消费，原108CTA下最多48个第二段等待，其余CTA继续推进。

R1过度保守地把第二段MMA也串行化，有限正确但略慢。R2纠正该依赖，并在任务全被领取后让空闲CTA进入出口屏障；两个调度改变的独立贡献未分离。最终整层相对同轮此前阶段版改善约0.33%–0.36%，两进程方向一致，幅度很小；相对原Torch1.1646/1.6808/2.7386×，不与上一轮比值直接相减。

debug确认提前消费，但最后生产和完成组均为3；部分工作提前没有移除最后组FFN1到FFN2的串行链。建议先量化这条路径及发布/等待成本，再考虑更细的段内流水。没有新增NCU或普通路径阶段profile，不把静态local1144B等同动态spill。没有声称消除ACT全局buffer或完成整层MegaMoE式warp调度。

9签名/36结构有限逐位及状态通过。R2目录错误导致的0调用FAILED、R1负结果和二进制恢复证明保留。两个研究入口采用R2，GPU已清理，四科学权限false；新GPU工作未启动。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[阶段表](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/STAGE-MAP.md)、[此前讨论](history/before-ffn-four-group-pipeline-20261022/DISCUSSION-SUMMARY.md)。
