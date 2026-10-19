# 独立执行路径与静态tile子图

本实现使用独立`wam_optimized_block_cuda`动态库。`block.py`、`parameters.py`、`native.py`和C++执行器均位于optimized，不继承或链接baseline执行器。旧`attention.py/build.py`和旧attention/FFN动态库是冻结baseline的依赖，保持原样；新构建只使用`build_subgraph.py`。

## 三种执行配置

```python
from wam_ops.install import install_model

# 独立顺序实现：作为optimized内部的等价起点，默认配置。
with install_model(model, 'optimized', execution_mode='sequential'):
    output = model(...)

# self/cross attention从显式任务坐标调用Flash设备tile。
with install_model(model, 'optimized', execution_mode='tiled'):
    output = model(...)

# 独立消融：Q16另启用FFN1 tile＋GELU，Q128保持顺序FFN。
with install_model(model, 'optimized', execution_mode='tiled', fused_ffn1=True):
    output = model(...)
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

`csrc/attention_tile.cuh`提供显式`AttentionTile{batch,head,query_block,kv_split,kv_splits}`的device入口。任务内完成QK、mask、在线softmax与PV；保留原Flash迭代顺序。标准分支query128/KV64，split分支query64/KV128，仍按原启发式选择split数并用原combine归并。`attention_tiles.cu`将原三维CTA网格解码成显式坐标，一CTA执行一个任务；没有persistent循环、原子领任务或跨阶段调度。

`csrc/ffn_tile.cuh`从固定CUTLASS派生显式`GemmTile{row,column}`设备函数，调用MMA mainloop与epilogue。Q16 FFN1的CTA尺寸64×128×64；保留BF16 GEMM舍入后再GELU。仅允许非split-K的既有FFN1形状。来源与许可证保留。

QKV/Wo与FFN2目前仍是直接cuBLASLt GEMM；没有声称这些已经成为可由persistent worker直接调用的tile。旧FFN2 CUTLASS数值失败仍有效，不能以更宽容精度门替换它。下一阶段须在独立优化版解决设备化边界，或明确保留分阶段库调用。

## 本轮边界

- 完整原生baseline及其共享依赖冻结；新源码、编译目录和动态库独立。
- 当前static tile入口可作为后续persistent worker的基础，不等于已实现MegaTransformer调度。
- 每候选action仍Q16，B14不改成可以相互注意的Q112；当前仍用原candidate pack路径，没有暗中换成G1/G2共享prefix算法。
- 数值验证与性能分开；共享GPU验证的墙钟不作加速证据。
- 已测范围及失败以[M31实际结果](../../../.pipeline/experiments/transformer-tile-subgraph-20261019/RESULT.md)为准；少量输入通过不证明所有输入、完整机器人质量或E2E。

构建：在本目录执行`MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build_subgraph.py build_ext --build-temp build_subgraph/temp --build-lib build_subgraph/lib --inplace`。不会覆盖旧baseline或旧共享DSO。
