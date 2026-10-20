# WAM 单层 Transformer 算子

当前入口：[optimized/whole_layer.py](wam_ops/optimized/whole_layer.py) → [whole_layer.cu](wam_ops/optimized/csrc/whole_layer.cu)。完整一层在一次 cooperative persistent kernel 内执行，包含 self/cross attention、所有投影、FFN、Norm、RoPE、残差与KV管理；不做30层大融合。

[最终有限结果](../.pipeline/experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)：9种真实输入×阶段/行组ready两路径，输出及全部可配对中间边界逐位一致；每路径18个合法有效KV转移直接对原Torch通过。默认阶段路径仍慢native2.68–5.02倍，ready又慢1.2%–6.0%。**有限精度通过，性能NO-GO，production_admitted=false。**

## 两份实现与源码

只维护baseline和optimized。baseline及vendor依赖冻结，原Torch模型只读；旧M28/M31/M32工具和原失败保留，但不作为当前整层单kernel入口。

|位置|作用|
|---|---|
|`wam_ops/baseline/{block.py,native.py,csrc/block.cu}`|冻结完整顺序基线，GEMM复用cuBLASLt、attention复用固定设备实现|
|`wam_ops/optimized/whole_layer.py`|当前参数/缓存绑定和有限形状契约|
|`wam_ops/optimized/csrc/whole_layer.cu`|完整单层DAG、队列、Norm/pointwise/KV与一次launch|
|`wam_ops/optimized/csrc/gemm_pipeline.cuh`|分工warp、cp.async双缓冲、WMMA及原BF16归约边界|
|`wam_ops/optimized/csrc/fa2_stage.cuh`、`fa2_load.cuh`|内部self/cross FA2、split/combine、共享history/私有suffix地址|
|`wam_ops/optimized/csrc/ffn_pipeline.cuh`|FFN1/GELU完整行组就绪后释放FFN2，bounded任务领取|
|`wam_ops/optimized/{fa2_codegen.py,norm_codegen.py}`|从冻结上游设备体生成内部FA2/LayerNorm|

结构/线程/构建见[WHOLE_LAYER.md](wam_ops/optimized/WHOLE_LAYER.md)。旧顺序/静态子图路径、安装器和工具的语义见[历史README](../.pipeline/experiments/single-layer-async-gemm-20261020/README-before-final-followup.md)，不能将旧安装入口误当作当前单kernel实现。

## 调用

```python
from wam_ops.optimized.whole_layer import WholeLayerBlock
with torch.inference_mode():
    kernel = WholeLayerBlock(block)  # 默认 tile_pipeline=False
    output = kernel(*args, **kwargs)
```

`tile_pipeline=True`仅为保留的负结果研究对照。默认不替换生产模型。真实范围：SM80、固定Torch2.9提交，D3072/H24/head128/FFN14336，B2 Q16/Q128或B14 Q16，cross512及已登记短中长KV；不是任意shape或训练API。

B14是7候选×CFG2，每个attention实例仍Q16，不跨candidate共享private KV。少量fixture、单block replay和一个kernel不等于完整verifier/机器人成功率或E2E收益。四项质量权限仍false；最强cache/Graph及完整1/3/7矩阵未完成。
