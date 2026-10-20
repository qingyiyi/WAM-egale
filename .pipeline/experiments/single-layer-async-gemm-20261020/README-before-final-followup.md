当前自有[整层单kernel入口](wam_ops/optimized/WHOLE_LAYER.md)已接入内部FA2 self/cross与split-combine；18同输入attention逐位PASS，完整block仍数值NO-GO、慢于native。最新[速度/误差](../.pipeline/experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)。以下native/M31/M32段落描述冻结对照及历史阶段；当前修改仅在optimized。

# WAM完整block算子研究

当前目录约束：[冻结baseline，仅在optimized推进tile优化](../.pipeline/memory/global/requirements/decision-冻结baseline并仅在optimized实现tile优化.md)。M31已经建立独立optimized参数/执行器及`wam_optimized_block_cuda`构建产物，不继承或链接baseline。旧attention/FFN源码与动态库仍为baseline冻结依赖，不修改或覆盖。实际实现与三种开关见[子图说明](wam_ops/optimized/SUBGRAPHS.md)。

当前设计修订：[self-attention、cross-attention与FFN的完整tile子图](../.pipeline/experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)。M32已实现单层内attention/FFN1常驻CTA tile循环，9真实block/90有效状态/48原语检查逐位通过；两进程探索计时却比静态路径更慢，B14约4.42%–6.26%。见[M32完整结果](../.pipeline/experiments/persistent-tile-remote-20261019/RESULT.md)。当前只研究一层完整block；30层不是本轮交付门。

最新协作交接：[真实结构/9签名/27参数与完整源码路径](../.pipeline/experiments/transformer-tile-handoff-20261019/HANDOFF.md)、[综合HW的tile/persistent设计](../.pipeline/experiments/transformer-tile-handoff-20261019/DESIGN.md)、[工作依赖与止损](../.pipeline/experiments/transformer-tile-handoff-20261019/WORK-OUTLINE.md)、[MegaMoE固定源码映射](../.pipeline/experiments/transformer-tile-handoff-20261019/MEGAMOE-NOTES.md)。M30是历史CPU设计；M32新增阶段内部persistent实现，尚无跨阶段ready队列或warp专职分工。后续优化限定optimized。

只维护`wam_ops/baseline`与`wam_ops/optimized`两份实现。M28按用户新要求把完整forward接到独立C++/CUDA；投机源码、配置、原tile和生产默认不变，原Torch模型是只读oracle。目标仅冻结推理，须在no_grad/inference_mode下调用，不提供autograd。

## 从这里读源码

| 文件 | 作用 |
|---|---|
| `wam_ops/baseline/block.py` | 原参数/state_dict布局；完整forward进入NativeBlock |
| `wam_ops/baseline/native.py` | 绑定27参数、7个eps与当前cache，调用原生入口 |
| `wam_ops/baseline/csrc/block.cu` | 完整顺序block、Norm/RoPE/调制/残差、KV写入/驱逐/打包/恢复 |
| `wam_ops/baseline/csrc/norm.cuh` | 固定Torch的LayerNorm/RMSNorm CUDA原语，保留许可证 |
| `wam_ops/baseline/csrc/gemm.h` | 直接cuBLASLt；复现原Torch descriptor/bias/workspace/heuristic |
| `wam_ops/optimized/block.py` | 独立参数/API容器，默认等价顺序路径 |
| `wam_ops/optimized/native.py` | 独立动态库绑定，sequential/tiled/persistent及FFN1开关 |
| `wam_ops/optimized/csrc/block.cu` | 实际self/cross/FFN三段子图，保留原算术边界 |
| `wam_ops/optimized/csrc/attention_tile.cuh` | 显式query tile坐标的Flash设备函数 |
| `wam_ops/optimized/csrc/persistent.cuh` | 常驻CTA固定步长任务循环与共享内存复用屏障 |
| `wam_ops/optimized/csrc/ffn_tile.cuh` | 显式行/列tile坐标的FFN1 MMA与epilogue |
| `wam_ops/optimized/csrc/attention.cu` | 冻结baseline依赖；新子图使用attention_tiles.cu |
| `wam_ops/optimized/csrc/ffn.cu` | 冻结旧FFN原语；新子图使用ffn_tiles.cu |
| `wam_ops/install.py` | 实例安装/还原，参数与cache原位共享 |

顺序：FP32调制表→norm1/调制→QKV→RMSNorm/RoPE→合法历史/私有KV→self attention/投影→门控残差→norm2→cross QKV/attention/投影→残差→norm3/调制→FFN1/GELU/FFN2→门控残差。

Linear/Norm子模块保存兼容参数，完整forward不调用其Torch数学forward。Tensor句柄、allocator、stream、cuBLAS handle/workspace仍依赖PyTorch运行时；GEMM复用cuBLASLt，未声称自研GEMM。完整block由多个CUDA kernel顺序执行，不是一整个kernel或30层融合。

当前支持SM80、D3072、24heads×128、Q16 action/Q128 video、original KV与只读candidate_rows。保留BF16计算及原FP32调制表/Norm参数。B14是7候选×CFG2，每个attention实例仍Q16，不跨候选读取私有KV。驱逐验证针对真实2160容量，不外推到所有形状/训练。

## 使用与构建

```python
import torch
from wam_ops.install import install_block
with torch.inference_mode(), install_block(block, 'optimized', execution_mode='tiled'):
    output = block(*args, **kwargs)
```

当前工作区baseline及旧依赖已构建并冻结，后续只构建独立优化库：

```bash
cd /rjs/cxz/论文探索/算子研究/wam_ops/optimized
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python build_subgraph.py build_ext --build-temp build_subgraph/temp --build-lib build_subgraph/lib --inplace
```

通过`install_block(block, 'optimized', execution_mode='tiled', fused_ffn1=True)`启用tile路径；默认`sequential`且FFN1融合关闭。`execution_mode='persistent'`启用新常驻CTA研究路径，当前不推荐作为加速默认值。Q128不启用FFN1融合。本机旧M31 DSO需重编才能运行新ABI，远端新DSO已验证。旧`build.py`供冻结版本追溯，不作为当前优化构建入口。

绑定Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`；本地CUDA12.4、wheel CUDA12.6差异明确保留，不启fast-math。CUTLASS固定v3.5.1沿用项目缓存，可用`python -m tools.fetch_cutlass`恢复；vendor是依赖，不是第三份WAM实现。

## 验证与性能

当前单层入口为`tools.verify_subgraphs --standalone --include-persistent`、`tools.audit_persistent`与`tools.time_persistent`，均须独立登记和新输出目录。全量结果见M32链接。以下为历史工具：`tools/verify_native.py`核对关键中间量/原语；`tools/verify_native_states.py`核对Q128和write1→write2→read0；`tools/replay.py --correctness-only`核对实际30层、安装还原及参数地址；`tools/time_native.py`测完整动态block，包含KV管理/打包/同步。

计时自然空闲，original_A/B做A/A，baseline/optimized逐轮交错，两个独立进程全部原始值保留；共享GPU只做精度。动态KV需要host同步，目前未支持CUDA Graph，不与预缓存Graph混比，不称新candidate-aware调度收益。

30层B14为保存的真实verifier；B2为同root CFG0/7行派生诊断，不能称原始顺序B2轨迹。原始B2/B14完整block及完整canonical video请求另有直接覆盖。冻结请求重放不等于机器人episode、成功率或任务E2E。

M25–M27旧工具/结果属于旧attention替换、prepared/Graph和FFN原语路径，不套用到当前native；旧源码已归档。旧CUTLASS FFN2逐位失败保留，当前FFN2采用原Torch配置cuBLASLt。所有旧FAILED/资源中断/性能门不改。

[当前M28完整结果](../.pipeline/experiments/block-native-baseline-20261019/RESULT.md)、[完整路径与源码讲解](../.pipeline/experiments/block-native-baseline-20261019/CODE-GUIDE.md)、[九点双进程动态block性能](../.pipeline/experiments/block-native-baseline-20261019/PERFORMANCE-TABLE.md)。六actual30 action query、原参数grad flags、全部声明有效状态有限逐位通过；未比较最强B14/Graph或测量新chunk/E2E。四项投机质量/TEST/非劣/正式性能权限仍false；研究替换不是生产放行。

M31实现与实际验证见[结果](../.pipeline/experiments/transformer-tile-subgraph-20261019/RESULT.md)。M31静态tile子图与M32新persistent证据分列；QKV/Wo/FFN2仍用cuBLASLt，不把旧M28完整模型结论套给新单层代码。

新子图验证入口是`python -m tools.verify_subgraphs`（必须提供登记与新输出目录）。旧`verify_native`的`--optimized`仍指M28共用执行链，不作为M31验证入口。
