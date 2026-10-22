# 当前执行：FFN写回融合精度通过，性能小幅退化

FFN1+GELU、FFN2+门控残差3已直接实现在optimized。9真实签名/36结构条件的有限逐位及输入/有效KV/RNG验证通过；两机研究入口采用融合版，旧源码/DSO保留，本机未另做GPU计时。

|KV|同轮融合前 / 原Torch加速比|融合后 / 原Torch加速比|
|---|---:|---:|
|288|1.1773×|1.1646×|
|1152|1.7085×|1.6893×|
|2160|2.7673×|2.7445×|

表内加速比定义均为T_Torch/T_自有。两独立进程、远端A100-SXM4、相同真实输入及最终Cross K/V缓存，warm完整单层CUDA Graph。融合本身比旧自有慢1.09%/1.14%/0.83%；不能称为融合提速，更不是30层或机器人E2E。后续唯一外部对照为原Torch；旧guarded证据保留在原报告。

独立GELU1568任务、残差336任务消失，warm逻辑任务3400/3472/3400，grid31→29；数学工作保留。当前MMA与GELU写回仍顺序执行，没有新专职warp或双缓冲重叠。未做本版逐阶段profile，慢因仍待分解。

首CPU编译变量遮蔽失败保留，修复后同截止完成。17:37:30 owned清理，累计8948.557/36000秒，原本轮18:12:32截止不变；无本轮GPU任务运行。四科学权限false、production_admitted=false，旧paused72及全部失败不改。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)、[阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[HW](../.pipeline/PROGRESS.compact.md)、[优化索引](../.pipeline/experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。

下一步建议先隔离两处融合增量，再做MMA/epilogue流水；尚未启动该新作业。历史RoboTwin状态沿原报告，本轮未恢复其过期队列。
