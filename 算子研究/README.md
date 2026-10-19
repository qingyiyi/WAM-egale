# WAM完整block算子研究

只维护`wam_ops/baseline`与`wam_ops/optimized`两份实现。M28按用户新要求把完整forward接到独立C++/CUDA；投机源码、配置、原tile和生产默认不变，原Torch模型是只读oracle。目标仅冻结推理，须在no_grad/inference_mode下调用，不提供autograd。

## 从这里读源码

| 文件 | 作用 |
|---|---|
| `wam_ops/baseline/block.py` | 原参数/state_dict布局；完整forward进入NativeBlock |
| `wam_ops/baseline/native.py` | 绑定27参数、7个eps与当前cache，调用原生入口 |
| `wam_ops/baseline/csrc/block.cu` | 完整顺序block、Norm/RoPE/调制/残差、KV写入/驱逐/打包/恢复 |
| `wam_ops/baseline/csrc/norm.cuh` | 固定Torch的LayerNorm/RMSNorm CUDA原语，保留许可证 |
| `wam_ops/baseline/csrc/gemm.h` | 直接cuBLASLt；复现原Torch descriptor/bias/workspace/heuristic |
| `wam_ops/optimized/block.py` | 复用native baseline，仅Q16启用FFN1＋GELU |
| `wam_ops/optimized/csrc/attention.cu` | 独立Flash CUDA，不转调Torch SDPA |
| `wam_ops/optimized/csrc/ffn.cu` | FFN1＋GELU CUTLASS原语，保留两次BF16舍入 |
| `wam_ops/install.py` | 实例安装/还原，参数与cache原位共享 |

顺序：FP32调制表→norm1/调制→QKV→RMSNorm/RoPE→合法历史/私有KV→self attention/投影→门控残差→norm2→cross QKV/attention/投影→残差→norm3/调制→FFN1/GELU/FFN2→门控残差。

Linear/Norm子模块保存兼容参数，完整forward不调用其Torch数学forward。Tensor句柄、allocator、stream、cuBLAS handle/workspace仍依赖PyTorch运行时；GEMM复用cuBLASLt，未声称自研GEMM。完整block由多个CUDA kernel顺序执行，不是一整个kernel或30层融合。

当前支持SM80、D3072、24heads×128、Q16 action/Q128 video、original KV与只读candidate_rows。保留BF16计算及原FP32调制表/Norm参数。B14是7候选×CFG2，每个attention实例仍Q16，不跨候选读取私有KV。驱逐验证针对真实2160容量，不外推到所有形状/训练。

## 使用与构建

```python
import torch
from wam_ops.install import install_model
with torch.inference_mode(), install_model(model, 'baseline'):  # 或 'optimized'
    output = model(input_dict, **kwargs)
```

在项目va环境先构建attention、FFN底层组件，再构建完整block：

```bash
cd /rjs/cxz/论文探索/算子研究/wam_ops/optimized
WAM_BUILD_TARGET=attention MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build.py build_ext --inplace
WAM_BUILD_TARGET=ffn MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build.py build_ext --inplace
cd /rjs/cxz/论文探索/算子研究/wam_ops/baseline
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build.py build_ext --inplace
```

绑定Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`；本地CUDA12.4、wheel CUDA12.6差异明确保留，不启fast-math。CUTLASS固定v3.5.1沿用项目缓存，可用`python -m tools.fetch_cutlass`恢复；vendor是依赖，不是第三份WAM实现。

## 验证与性能

`tools/verify_native.py`核对关键中间量/原语；`tools/verify_native_states.py`核对Q128和write1→write2→read0；`tools/replay.py --correctness-only`核对实际30层、安装还原及参数地址；`tools/time_native.py`测完整动态block，包含KV管理/打包/同步。

计时自然空闲，original_A/B做A/A，baseline/optimized逐轮交错，两个独立进程全部原始值保留；共享GPU只做精度。动态KV需要host同步，目前未支持CUDA Graph，不与预缓存Graph混比，不称新candidate-aware调度收益。

30层B14为保存的真实verifier；B2为同root CFG0/7行派生诊断，不能称原始顺序B2轨迹。原始B2/B14完整block及完整canonical video请求另有直接覆盖。冻结请求重放不等于机器人episode、成功率或任务E2E。

M25–M27旧工具/结果属于旧attention替换、prepared/Graph和FFN原语路径，不套用到当前native；旧源码已归档。旧CUTLASS FFN2逐位失败保留，当前FFN2采用原Torch配置cuBLASLt。所有旧FAILED/资源中断/性能门不改。

[当前M28完整结果](../.pipeline/experiments/block-native-baseline-20261019/RESULT.md)、[完整路径与源码讲解](../.pipeline/experiments/block-native-baseline-20261019/CODE-GUIDE.md)、[九点双进程动态block性能](../.pipeline/experiments/block-native-baseline-20261019/PERFORMANCE-TABLE.md)。六actual30 action query、原参数grad flags、全部声明有效状态有限逐位通过；未比较最强B14/Graph或测量新chunk/E2E。四项投机质量/TEST/非劣/正式性能权限仍false；研究替换不是生产放行。
