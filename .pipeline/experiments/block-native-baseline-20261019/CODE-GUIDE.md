# 从完整block入口读起

两份活跃实现均在 /rjs/cxz/论文探索/算子研究/wam_ops/，投机代码不改。按下面顺序读即可；原Torch只作工具层oracle。

1. **完整baseline入口**：/rjs/cxz/论文探索/算子研究/wam_ops/baseline/block.py:63。forward绑定NativeBlock后交给原生链；构造器保留原参数名称/state_dict布局。
2. **参数与缓存绑定**：/rjs/cxz/论文探索/算子研究/wam_ops/baseline/native.py:25。按原顺序绑定27个参数、7个eps、5个cache字段；detach只是共享内存只读别名，不复制权重、不改原requires_grad。要求no_grad/inference_mode。
3. **完整数学主链**：/rjs/cxz/论文探索/算子研究/wam_ops/baseline/csrc/block.cu:262。这是整个block，不只是attention。
4. **优化版入口**：/rjs/cxz/论文探索/算子研究/wam_ops/optimized/block.py:6。继承同一原生链，只开启Q16的FFN1＋GELU融合；其它步骤共用，避免维护两份重复数学。
5. **模型替换与还原**：/rjs/cxz/论文探索/算子研究/wam_ops/install.py:49。context manager逐个替换现有block的类与绑定，退出时恢复；不会把参数迁移或另建checkpoint。baseline和optimized均可安装全部30层。

## 一次forward做什么

block.cu:279先把输入按原边界转FP32做LayerNorm，再和sigma/time embedding组成的scale/shift调制，按原边界舍入为BF16。随后三次cuBLASLt线性层生成Q/K/V，Q/K用RMSNorm，RoPE按原实现的double计算后回到BF16。

block.cu:228的self_attention管理当前KV：按原mask/id/is_pred规则选择槽位、写入/驱逐、恢复read0临时mask，生成正确可见的K/V。B14的candidate_rows只是映射到各自CFG历史；private K/V不跨候选共享。16字节copy只加快字节搬运，CUB scan保留physical order。进入自有Flash CUDA后，每个逻辑attention仍为Q16；视频为Q128。

block.cu:287做self输出投影和带gate残差；289–295做原norm2、text cross attention及残差。296–309做第三次Norm/调制、FFN扩展、GELU、FFN压回和最终残差。baseline的两次FFN GEMM直接cuBLASLt；optimized在第一个GEMM后融合GELU，并保留原中间BF16舍入。

cuBLASLt接口在 /rjs/cxz/论文探索/算子研究/wam_ops/baseline/csrc/gemm.h。复制固定Torch的descriptor、bias、workspace和heuristic细节，避免“数学都是GEMM”但选择不同算法导致BF16不同。Norm的固定上游原语在同目录 norm.cuh，许可证在 LICENSE.pytorch。self/cross SDPA主入口在 /rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/attention.cu；融合FFN1在同目录 ffn.cu。

注意：自有实现仍依赖PyTorch Tensor/allocator/stream及成熟cuBLASLt/CUTLASS/固定Flash代码；不能称为从零实现全部库。当前支持的研究契约是SM80、D3072、24头×128、BF16及原FP32参数、Q16/Q128、真实KV容量2160；不支持训练或当前动态链的CUDA Graph。

## 如何看验证

- /rjs/cxz/论文探索/算子研究/tools/verify_native.py：六action、23关键中间量与六个原语逐位对照。
- /rjs/cxz/论文探索/算子研究/tools/verify_native_states.py：action/video的write1→write2→read0与真实驱逐。
- /rjs/cxz/论文探索/算子研究/tools/replay.py：实际模型安装全部30层、有效KV/RNG/参数/还原检查；--keep-original-grad-flags保留canonical参数标志。
- /rjs/cxz/论文探索/算子研究/tools/time_native.py：相同fixture、相同block参数、完整动态KV的四引擎轮换计时。安装在计时外；每轮输出、有效状态、输入和RNG亦核对。

源表、完整结果和局限见同目录 RESULT.md；504条原始计时和所有FAILED/PARTIAL保留。只有两份活跃实现；reference/是历史证据，不是第三个部署版本。
