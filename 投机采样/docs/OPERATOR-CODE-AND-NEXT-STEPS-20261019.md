> 此文记录重构前的目录和讲解。当前源码已迁入[算子研究](../../算子研究/README.md)，旧算子代码目录已删除，历史结果/源Git身份保留；投机侧代码不改。当前工作以[M25](../../.pipeline/experiments/operator-consolidation-20261019/PLAN.md)为准。

# 当前算子源码与后续工作

基于2026-10-19已完成的直接Torch复制版、固定科研源码ca5f37e，以及用户“先复现”的最新优先级。本页是源码导航和工作清单；具体结果以Experiment原件为准。

## 从哪里阅读

| 入口 | 内容与当前角色 |
|---|---|
| `experiments/direct_torch_block_copy_v1/block.py` | 当前直接复制基线；27行SDPA入口，338行attention forward，483行完整block forward |
| `experiments/direct_torch_block_copy_v1/binding.py` | 8行单block替换，42行模型各block替换；参数/cache原位保留，退出还原 |
| `experiments/direct_torch_block_copy_v1/gpu_replay.py` | 六真实fixture的完整输出、有效状态、输入/RNG和配对计时；不进入生产服务 |
| `experiments/block_sequential_baseline_v1/block_baseline.py` | 显式分阶段B0；有tap可观察中间输出，后续做依赖和阶段分析可复用 |
| `experiments/operator_torch_alignment_v2/csrc/standalone_flash_sdpa.cu` | 此前独立编译的SDPA CUDA入口，复用固定Torch的FlashAttention源码；不是整个block |
| `experiments/operator_torch_alignment_v2/integration.py` | 将上述CUDA SDPA安装到真实30层模型的self/cross attention，保留旧证据范围 |

主目录：`投机采样/experiments/direct_torch_block_copy_v1/`。当前完整block基线使用Torch/diffusers/einops；没有新的CUDA block融合或persistent scheduler。独立attention扩展是另一可选替换层，不是当前直接复制版默认路径。

## 顺着一次forward看计算

本地配置：30层，hidden3072，24头，每头128，FFN14336，cross-attention有LayerNorm，epsilon为1e-6。112维action latent经过前段映射之后，进入这里的是`[B,16,3072]`隐藏状态。

1. `WanTransformerBlock.forward`先把`temb.float()`与学习的scale/shift表相加，拆成self-attention及FFN各自的shift、scale、gate六组。它们调制当前去噪阶段的计算。
2. self-attention前对hidden转FP32做LayerNorm与调制，再转回输入dtype；送入`attn1`，随后按原精度顺序乘gate并加残差。
3. cross-attention以当前hidden作为Q、`encoder_hidden_states`作为K/V。这一路没有RoPE；输出按原有BF16残差加法累加。条件的更上游构造不在block内。
4. FFN前再次LayerNorm与调制，然后执行`3072→14336→3072`的两层投影，中间使用原approximate GELU及dropout模块。最后按原FP32运算与转回输入dtype的边界完成门控残差。

attention内部：三组Linear产生Q/K/V；Q/K各做RMSNorm；拆成24头；self-attention按源码用FP64/complex计算RoPE后转回原dtype。接着整理可见KV，调用`custom_sdpa`。该函数只转置到Torch要求的`[B,H,Q,D]`布局并调用`F.scaled_dot_product_attention`，返回后合并head、做输出投影。底层kernel由Torch后端执行，本模块没有重新实现softmax或矩阵乘法。

普通B2路径按原cache协议暂存或提交当前KV；`update_cache=0`调用后回收临时有效槽。候选B14路径通过`candidate_cache_rows`选中对应CFG历史，用`index_select`及`cat`拼接“该候选的历史＋该候选的私有16 token”，且不把候选写入持久历史。B14=2路CFG×7个候选，每个实例的Q仍是16，候选互相不可见。

短中长attention总K为288/1152/2160，其中历史为272/1136/2144、当前私有部分16。当前复制版仍有历史materialization，已有shared-prefix路径需要单独比较；不能把复制版当成已有最佳B14路径。

`copied_block`在forward之外切换block/attention类和SDPA函数，复用原参数、子模块与cache对象；它没有`return original_block(...)`这样的数学委托。`copied_model_blocks`遍历`model.blocks`安装，并用ExitStack还原。入口存在不表示已完成新版本完整30层验证。

## 已验证和缺口

直接复制版已完成：三个定义逐字/AST一致；六个真实action block0 fixture（B2/B14×短中长），18次数值检查逐位相同，全部有效KV/元数据、输入、当前CPU/CUDA RNG及历史GPU gold通过；计时后状态也通过。六点中位耗时比0.9677–1.0033，单轮比0.8587–1.0973，不能把更快点当优化收益。

本版本尚缺：完整30层安装/还原回归、相应完整policy必要检查、GPU合法cache写路径、视频block覆盖与更稳定计时。旧M22仅attention替换的full30/policy证据单独保留。保存fixture缺原生产CUDA stride/alias/RNG，不能声称恢复了这些字段。

原始结果：[直接复制版](../../.pipeline/experiments/direct-torch-block-copy-20261019/RESULT.md)。旧B0的66次GPU检查和旧A/A失败不修改。

## 后续工作清单

| 顺序 | 工作 | 完成交付/判断 | 当前状态 |
|---|---|---|---|
| 1 | 整模型复现收口 | 新复制版装入30层，复用已有真实B2/B14短中长query；核对最终输出、全部有效KV/metadata、RNG和还原；必要时覆盖完整policy的接受/提交/scheduler | 入口已具备，实际新版本完整模型回归待做 |
| 2 | 稳定基线计时 | 原版/复制版用同一组真实输入、相同工程选项；短中长block与完整verifier分别报，独占窗口交错执行；完整原始轮次与噪声保留 | 已有六点探索性数据，稳定计时待补 |
| 3 | 强B14基线与profile | 对齐已存在shared-prefix、O3、合法静态projection缓存/Graph；验证组合；量出GEMM、attention、norm/RoPE、copy、launch/host占比及实际shape | 组件已有，当前组合与归因待收口 |
| 4 | 一个子图的device原语与顺序实现 | 按profile只选择一个子图；可复用固定CUDA/CUTLASS实现。先构成P_seq，保持原BF16舍入和精度，核对原语性能，避免大GEMM先退化 | 尚无新的完整block设备原语/调度实现；候选未选 |
| 5 | 候选感知tile调度 | 同一设备原语P_sched对P_seq，再对最强B14；1/3/7状态×短中长。最多两个有依据配置，按实际tile/依赖/resource说明收益 | 条件触发，先完成1–4的证据 |
| 6 | 收益传递与论文证据 | block→30层verifier→完整chunk；记录sum(T_round)/sum(g)，再结合质量/任务成本；调度开关×共享缓存开关消融，严格区分机制和普通缓存收益 | 依赖候选/引擎冻结及登记质量权限 |

建议独占窗口顺利时：10/19–20优先完成1–3；10/21–22最多一个最小子图样机并go/no-go；10/23冻结候选/引擎，10/28主表。日期是执行目标，GPU预算按实际作业登记，不把旧截止或旧余额重置。首个子图可考虑FFN链或共享历史attention，但目前不据标题/直觉决定。

CPU可以并行准备源码/DAG、fixture来源核对、统计与报告；本机GPU顺序用于复现/测量，BAAI继续既定Object评估。正式计时与profiler分开。若新调度相对强基线不能稳定获益，停止扩大；当前只恢复到基础复现工作，不自动开发大型30层融合。

四项质量/正式性能字段继续为false。系统有限机制测量不能放行被阻塞的TF正式45chunk/独立TEST。总体目标是质量约束下减少有效推进和任务成本，不以占用率或target调用数代替。
