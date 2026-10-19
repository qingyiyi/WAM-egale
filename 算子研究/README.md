# WAM算子研究

这里只维护两份实现：`wam_ops/baseline`和`wam_ops/optimized`。投机侧模型、配置、接受器、训练与评估代码保持原样。默认optimized仍只替换attention CUDA核心；新静态强基线和FFN设备原语必须通过各自实验门，不能自动装入生产模型。

```text
wam_ops/
  baseline/
    block.py        # 完整Transformer block：self/cross attention、FFN、残差
    attention.py    # 原Torch SDPA、RoPE与original KV协议
  optimized/
    block.py        # 复用baseline block，选择CUDA attention
    attention.py    # 延迟加载独立CUDA扩展
    prepared.py     # 六个真实fixture的静态KV/条件缓存强基线，独立admit
    ffn.py          # 显式安装FFN1+GELU原语，保留Torch FFN2
    csrc/attention.cu
    csrc/candidate_attention.cu  # 复用已有候选attention，只作强基线
    csrc/ffn.cu     # FFN设备原语研究；production_admitted=false
    csrc/vendor/    # 固定上游FlashAttention/CUTLASS源码与许可证
    build.py        # 按WAM_BUILD_TARGET构建一个组件
  install.py        # 模型实例的安装/还原，共享参数和缓存
tools/
  fixtures.py       # 读取现存真实block输入
  metrics.py        # 逐位、有效KV、输入/RNG比较
  replay.py         # 同输入完整block/实际30层验证与交错计时
  run.py            # 等待空闲GPU、运行两个独立进程并清理
  block_loader.py   # 只加载block0 checkpoint参数
  profile_block.py  # 六点强基线、逐位门、Graph与独立profile
  counter_block.py  # NCU真实完整block；重放/缓存口径另记
  device_ffn.py     # 独立中间量、整体FFN逐位及探索性计时
  bounded.py        # 生命周期上限、PID出生身份与自有进程清理
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

## 第1/2步的源码调用链

`baseline/block.py`保留原block顺序：norm1与调制→self attention→门控残差→norm2与cross attention→残差→norm3与调制→FFN→门控残差。`baseline/attention.py`负责Q/K/V投影、RMSNorm、RoPE、KV可见性、SDPA及输出投影。复制Torch路径的既有完整验证结果不由新原语改写。

`optimized/prepared.py`先在`PreparedBlock`中固定有效slot与原CFG行；`fill`用原B2/B14形状缓存固定条件的cross K/V投影，分别占12/84MiB。`activate`仅临时替换这个block的self-attention入口和两次固定投影，退出还原。B2保留原临时slot的物理排序与返回元数据；B14让每个Q16候选读取相同合法历史及自己的私有K/V。native与guarded为已有基线组件，缓存与Graph收益不称为MegaTransformer新贡献。

Graph只用于已admit且地址/内容固定的真实fixture。Python guard不在replay中重新执行；这里没有生产条件变化后的失效协议。每条新路径先核对输出、全部有效KV、全部metadata、输入及RNG，再测量。冷填充、Graph构建、profile和replay计时分列。

`optimized/csrc/ffn.cu`中，`linear_tile`在CUDA CTA内调用CUTLASS MMA/mainloop/epilogue，GELU表达式绑定原Torch版本。完整device FFN的FFN2没有逐位通过，保留为失败诊断。最终收缩的`RoundedGelu`先让原线性epilogue舍入到BF16，再在寄存器中做原GELU并再次舍入BF16，省去FFN1结果的global写回和GELU单独launch。`optimized/ffn.py`只临时替换`ffn.net[0]`，`ffn.net[2]`仍是原Torch FFN2。

这就是本次的“融合”：将FFN1和GELU放在同一CTA epilogue中，保留二者之间的数值舍入；没有persistent任务队列或完整FFN/30层融合。`tools/device_ffn.py`保存完整device FFN的失败与分项结果；`tools/verify_ffn1.py`对最终收缩组件做中间量和完整block门检、独立进程交错计时。显式研究入口不改变`install_model`的默认optimized路径。

六点实际线性层M=32/224，FFN1的N/K=14336/3072，FFN2的N/K=3072/14336；这已由实际addmm记录核对。attention每个候选仍为Q16。完整结果、未通过点与源身份见[本轮HW](../.pipeline/experiments/operator-profile-device-20261019/RESULT.md)。

构建设备原语依赖时先运行`python -m tools.fetch_cutlass`，仅把官方固定CUTLASS v3.5.1 headers放到项目内`.pipeline/cache`；再在optimized下设置`WAM_BUILD_TARGET=ffn`构建。候选基线用`WAM_BUILD_TARGET=candidate`，默认attention不变。原模型大文件不做独立整套hash。

本轮有限验证已完成：两个进程覆盖六点block/实际30层及write1/2，所有输出/有效状态逐位通过。完整计时数据已保存，但三次资源检查发现外来GPU进程，稳定性能仍PENDING；登记的150次full30调用用满，自有GPU worker均退出。[最终结果与全部原始表](../.pipeline/experiments/operator-consolidation-20261019/RESULT.md)。

用户再次要求空闲继续后，新有界接续两进程完成108项逐位检查，完整模型baseline/原版中位耗时0.9539–1.0058、optimized0.9535–1.0218；432full30/444block后自有退出。102次采样无外来PID，但NVML空卡利用率异常和严格±2.5%全点门仍未解决，不签正式性能。[新全部结果](../.pipeline/experiments/operator-stable-timing-20261019/RESULT.md)。
