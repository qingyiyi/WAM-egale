# WAM 单层 Transformer 算子

活动优化入口为简易版[whole_layer.py](wam_ops/optimized/whole_layer.py)和[whole_layer.cu](wam_ops/optimized/csrc/whole_layer.cu)。完整一层在一个persistent kernel中运行，覆盖self/cross attention、投影、FFN、Norm、RoPE、残差与KV管理。

|位置|作用|
|---|---|
|`wam_ops/baseline/`|冻结顺序原生基线及原Torch对照|
|`wam_ops/optimized/whole_layer.py`|当前简易版参数与cache绑定|
|`wam_ops/optimized/csrc/whole_layer.cu`|完整单层DAG、tile队列、Norm/pointwise与launch|
|`wam_ops/optimized/csrc/cutlass_gemm.cuh`|action/context的大M设备GEMM|
|`wam_ops/optimized/csrc/gemm_pipeline.cuh`|小M加载/计算与BF16归约边界|
|`wam_ops/optimized/csrc/fa2_stage.cuh`|内部self/cross FA2与合法KV寻址|

源码讲解与构建见[WHOLE_LAYER](wam_ops/optimized/WHOLE_LAYER.md)，当前精度、性能与逐阶段结果见[本轮报告](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)。历史复杂版留在实验归档；baseline、投机侧与原tile不改。只维护当前简易整层优化路径。

9种真实输入的有限输出/23中间量/有效KV/重复/输入/RNG逐位检查已通过。阶段仍依次执行并grid.sync；尚不是跨阶段ready执行器。B14每实例Q16，private KV独立。单层工程测量不能代替强cache/Graph、完整verifier、机器人质量与E2E；四质量权限仍false。
