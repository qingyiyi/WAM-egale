# SM120 FFN最新有限接续：NO_GO，保留A

读取本文件所在固定提交，不跟随main覆盖旧科学身份。本轮基线来自3eb0026705e603d6268731bb26b34261acca696c，现场冻结源与该blob相同；不是旧A100319ac6a。

T0数值与旧负结果、T1成本、唯一PAIR2048x2设计及T3实际完成。相邻两段同CTA减少global partial交接，但保持七段独立FP32 MMA和七次BF16舍入，三真实block0 fixture有限逐位/状态通过；无插桩两进程三KV整层慢6.25%–7.89%，同轮FFN慢约18%–19%。不采用，当前SELF-QKV-TMA-4MMA源码/DSO未改。

- [最终结果与五问](../.pipeline/experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)
- [完整17阶段表、tile/warp/同步及Torch/Guarded](../.pipeline/experiments/sm120-ffn-next-20261024/STAGE-TABLE.md)
- [数值递推](../.pipeline/experiments/sm120-ffn-next-20261024/NUMERIC-CONTRACT.md)
- [成本与决定](../.pipeline/experiments/sm120-ffn-next-20261024/BOTTLENECK-AND-DECISION.md)
- [固定源码与原始数据入口](../.pipeline/experiments/sm120-ffn-next-20261024/SOURCE-AND-RESULT-ENTRY.json)
- [原始标量与采用决定](../.pipeline/experiments/sm120-ffn-next-20261024/RESULTS.json)
- [旧负结果矩阵](../.pipeline/experiments/sm120-ffn-next-20261024/NEGATIVE-EVIDENCE.md)
- [实际资源与清理](../.pipeline/experiments/sm120-ffn-next-20261024/RESOURCE-AND-STOP-REPORT.md)
- [RoboTwin当前现场快照](../.pipeline/experiments/sm120-ffn-next-20261024/ROBOTWIN-PROGRESS.json)

最强共同边界：5090/SM120、B14/Q16/block0、三KV288/1152/2160、warm CrossKV、全部eager/no Graph。当前A相对原Torch完整单层约1.251/1.758–1.761/2.191–2.201×；对Guarded约1.110/1.084–1.089/1.047×。这些是单层重放，不是完整模型或机器人E2E。七段有限逐位不等于所有输入已证明。

T1前序等待约0.2μs/任务、MMA与输入/结果包络约25.8μs；不能把CTA累计包络当GPU关键路径百分比。A/B资源相同，不支持“shared变大导致退化”。任务变粗损失并行性是待分离假设，真实DRAM/L2/动态spill未知。停止本候选，不继续tile/warp搜索；建议下一轮先解释MMA/输入包络。

工程质量、独立TEST、非劣、正式性能、生产准入五项仍false。5090本轮自有进程已退出，实例仍开。RoboTwin双A100继续固定DEV队列，见带时间戳快照；pair406/T因VAE显存OOM中断，新namespace只接未开始组，未自动重试该T。已结束C/P自然失败保留，不能据它们补造完整组三臂。

公开省略原始tensor/KV、七段P/R的.pt、模型权重、DSO及SSH部署runner；原件在本机和既有5090实验目录。公开源码/patch/原始标量JSON/失败/收据可审查，镜像不是含全部资产的自包含环境。
