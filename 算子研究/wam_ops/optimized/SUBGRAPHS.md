# 单层Transformer block：tile与persistent执行

本实现使用独立`wam_optimized_block_cuda`动态库。`block.py`、`parameters.py`、`native.py`和C++执行器均位于optimized，不继承或链接baseline执行器。旧`attention.py/build.py`和旧attention/FFN动态库是冻结baseline的依赖，保持原样；新构建只使用`build_subgraph.py`。

当前实现与主要验收范围是一层完整block，非30层融合。最终模型E2E后续在本机验证。

## 执行配置

```python
from wam_ops.install import install_block

# 独立顺序实现：作为optimized内部的等价起点，默认配置。
with install_block(block, 'optimized', execution_mode='sequential'):
    output = block(...)

# self/cross attention从显式任务坐标调用Flash设备tile。
with install_block(block, 'optimized', execution_mode='tiled'):
    output = block(...)

# 独立消融：Q16另启用FFN1 tile＋GELU，Q128保持顺序FFN。
with install_block(block, 'optimized', execution_mode='tiled', fused_ffn1=True):
    output = block(...)

# attention与FFN1使用常驻CTA循环；其余阶段依赖保持顺序。
with install_block(block, 'optimized', execution_mode='persistent', fused_ffn1=True):
    output = block(...)
```

调用仍须位于`torch.inference_mode()`中。参数、调制精度、KV可见性、全部原舍入点不变。baseline不接受上述优化参数。默认等价路径不自动启用已有FFN融合。

## 可执行子图

```text
self_subgraph:
  全行LN/调制 → QKV GEMM → 全D Q/K RMSNorm/RoPE
  → 原KV管理 → attention输出tile → Wo GEMM → gate/residual
cross_subgraph:
  LN → Q投影/全D RMSNorm；context K/V投影及K归一化
  → attention输出tile → Wo GEMM → residual
ffn_subgraph:
  全行LN/调制 → FFN1[可选tile＋GELU] → FFN2 GEMM → gate/residual
```

三段位于`csrc/block.cu`，实际forward调用它们。当前同一CUDA stream按阶段提交，消费者等待整个生产阶段完成。归约完整行、全部head输出、私有KV就绪均由这个严格顺序满足。尚未用局部ready令牌提前释放消费者。

`csrc/attention_tile.cuh`提供显式`AttentionTile{batch,head,query_block,kv_split,kv_splits}`的device入口。任务内完成QK、mask、在线softmax与PV；保留原Flash迭代顺序。标准分支query128/KV64，split分支query64/KV128，仍按原启发式选择split数并用原combine归并。静态`tiled`模式一CTA一个任务；`persistent`模式用同一设备函数循环处理多个任务。

`csrc/ffn_tile.cuh`从固定CUTLASS派生显式`GemmTile{row,column}`设备函数，调用MMA mainloop与epilogue。Q16 FFN1的CTA尺寸64×128×64；保留BF16 GEMM舍入后再GELU。仅允许非split-K的既有FFN1形状。来源与许可证保留。

QKV/Wo与FFN2目前仍是直接cuBLASLt GEMM；没有声称这些已经成为可由persistent worker直接调用的tile。旧FFN2 CUTLASS数值失败仍有效，不能以更宽容精度门替换它。下一阶段须在独立优化版解决设备化边界，或明确保留分阶段库调用。

## 常驻CTA如何工作

`csrc/persistent.cuh`定义共同任务循环：CTA编号为c时，依次处理`c, c+gridDim.x, c+2*gridDim.x, ...`。默认CTA数为`min(任务数, SM数量)`，每个CTA的warp协作完成一个tile。任务集合按同余类互斥且覆盖全部任务，不使用全局原子领取，也不依赖其他CTA先运行，因此没有跨CTA自旋死锁。

每个tile结束后执行`cp.async.wait_all`与CTA barrier，确认异步拷贝结束、所有线程离开前一个tile，再复用共享内存。Flash的局部early-return按task统一，不跳过调用方的复用barrier。FFN每次重建迭代器并清零accumulator，不把上一tile累加值带入下一tile。

`attention_tiles.cu::persistent_attention`服务self/cross；`ffn_tiles.cu::persistent_linear`服务可选FFN1＋GELU。均保留原tile归约/舍入，模式0/1/2分别为sequential/tiled/persistent。审计接口单独记录逐task访问次数和每CTA完成数，计时不启用这些审计写入。

这是阶段内部的常驻CTA实现。没有跨self/cross/FFN的ready队列，没有不同warp专职加载/计算，没有整个block单kernel融合，也未实现跨候选公共KV的新复用。第一版可执行机制不自动构成MegaTransformer创新或性能收益。

## 本轮边界

- 完整原生baseline及其共享依赖冻结；新源码、编译目录和动态库独立。
- 当前persistent仍为阶段内部静态任务分配；候选感知DAG调度和warp角色分工尚待实现。
- 每候选action仍Q16，B14不改成可以相互注意的Q112；当前仍用原candidate pack路径，没有暗中换成G1/G2共享prefix算法。
- 数值验证与性能分开；共享GPU验证的墙钟不作加速证据。
- M31静态实现证据保留；persistent新版本以[M32](../../../.pipeline/experiments/persistent-tile-remote-20261019/RESULT.md)的实际结果为准，不能继承旧通过结论。有限输入不证明完整机器人质量或E2E。

构建：在本目录执行`MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build_subgraph.py build_ext --build-temp build_subgraph/temp --build-lib build_subgraph/lib --inplace`。不会覆盖旧baseline或旧共享DSO。

本机旧M31独立DSO未自动重建；Python入口检查persistent ABI，遇旧二进制会明确要求重编，不静默把persistent当作静态模式。当前新版本在BAAI隔离目录构建，远端结果与本机最终E2E分列。

M32结果：9真实完整block、90连续有效KV状态与48原语覆盖逐位通过；两独立进程中，persistent+FFN1比静态tile+FFN1在B14慢4.42%–6.26%。默认仍保持sequential，保留静态路径；没有新增persistent加速主张。
