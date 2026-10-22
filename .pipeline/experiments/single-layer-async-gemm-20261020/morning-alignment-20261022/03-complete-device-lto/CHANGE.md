# 03：仅补齐compile LTO

原CUDA源码全部恢复11。build_whole_layer.py仅新增-dlto及解释注释。原rdc/dlink保留；不变fast_math/FMA/BF16边界。实际选项/IR与静态callee计数核查后，按原有限精度门及三KV两进程判断；非微小注释收益。详见本段前瞻修订1。
