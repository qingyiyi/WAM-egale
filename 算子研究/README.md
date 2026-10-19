# WAM算子研究

这里只维护两份实现：`wam_ops/baseline`和`wam_ops/optimized`。投机侧模型、配置、接受器、训练与评估代码保持原样。现阶段优化候选只替换attention CUDA核心，完整block按原顺序执行。

```text
wam_ops/
  baseline/
    block.py        # 完整Transformer block：self/cross attention、FFN、残差
    attention.py    # 原Torch SDPA、RoPE与original KV协议
  optimized/
    block.py        # 复用baseline block，选择CUDA attention
    attention.py    # 延迟加载独立CUDA扩展
    csrc/attention.cu
    csrc/vendor/    # 固定上游FlashAttention/CUTLASS源码与许可证
    build.py        # 单文件扩展构建
  install.py        # 模型实例的安装/还原，共享参数和缓存
tools/
  fixtures.py       # 读取现存真实block输入
  metrics.py        # 逐位、有效KV、输入/RNG比较
  replay.py         # 同输入完整block/实际30层验证与交错计时
  run.py            # 等待空闲GPU、运行两个独立进程并清理
```

## 先看哪几处

先读`baseline/block.py`的`TransformerBlock.forward`，再读`baseline/attention.py`的`Attention.forward`。后者执行Q/K/V投影、Q/K归一化、RoPE、历史KV与私有KV整理、SDPA、输出投影。

`optimized/block.py`继承同一数学流程，只把attention操作指向`optimized/attention.py`的`cuda_sdpa`；CUDA入口在`csrc/attention.cu`。没有重复复制整套block代码。当前支持原Torch模型、original KV协议、A100/SM80、BF16、head_dim128；attention底层不是完整block融合。

安装已有模型：

```python
from wam_ops.install import install_model

with install_model(model, "baseline"):   # 或 "optimized"
    output = model(input_dict, **kwargs)
```

context在forward之外切换类和attention入口。参数、cache对象和buffer地址保留，退出还原原类和原操作。输入模型的attention后端需为torch、KV协议为original，不与其他实验forward覆盖混装。

## 构建与验证

在原`va`环境、本目录下构建：

```bash
cd wam_ops/optimized
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build.py build_ext --inplace
```

编译保留原对齐版的精度选项，未启用fast-math。vendor是必要第三方依赖，不属于第三份WAM实现。

验证与计时共用`tools/replay.py`，读取现有真实tensor，不启机器人模拟器。实际作业登记、原始数据、失败/清理与汇总都保存在[HW本项](../.pipeline/experiments/operator-consolidation-20261019/PLAN.md)。正式耗时需无GPU争用，profiler单独运行。

现有B14 full30是保存的真实verifier输入；B2 full30诊断明确使用同root的CFG0/7行与普通B2路径，不能称为保存的原始顺序B2轨迹。原始B2/B14完整block输入另有直接覆盖。每个计时输入先做同一进程的数值/状态检查；原始轮次与两个进程结果全部报告。

旧16个算子实现目录已移出工作区，308份源码可由登记的Git提交恢复，不在当前目录保留压缩副本。原tile仍被投机shared-prefix构建脚本引用，因此与投机依赖一同原位只读保留；原始实验结果保留。

当前结果以HW实际终态为准；文件整理或有限通过不改变投机质量/独立TEST/正式性能权限。
