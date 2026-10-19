# 直接复制Torch的完整block基线

`block.py`从固定`ca5f37e0482c2e38e3738e3d8e71e3b185990a23`的模型文件复制`custom_sdpa`、完整`WanAttention`（含cache助手）和`WanTransformerBlock`。三个定义不改计算源码；只收窄导入，支持原`attn_mode="torch"`路径。依赖Torch、diffusers、einops；没有自研CUDA、融合、静态projection缓存或G1/G2调度。

可独立构建`WanTransformerBlock(..., attn_mode="torch")`并严格载入原state_dict。替换现有模型时：

```python
from experiments.direct_torch_block_copy_v1.binding import copied_model_blocks

with copied_model_blocks(model):
    output = model(...)  # 原参数、norm/linear/FFN、缓存和Torch SDPA继续使用
```

替换完整block/attention类，退出时还原。绑定发生在forward以外，不增加数学包装调用；参数和cache不搬移。模型构建、scheduler、server都不在该模块中。本轮六fixture的验证不自动证明完整30层/机器人闭环或所有cache写路径。

数值与性能证据：[本轮登记](../../../.pipeline/experiments/direct-torch-block-copy-20261019/PLAN.md)。精度已通过的旧B0结果仍保留，新复制版未实测前不借用旧证书。
