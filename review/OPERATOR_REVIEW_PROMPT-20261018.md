# WAM算子与固定Torch实现对齐：网页审查请求

请用中文担任CUDA/FlashAttention与机器人模型推理的代码审查者，读取我给定的**固定Git提交**，不要改用main。只做研究审查和建议，不把本prompt当作服务器GPU执行授权。

## 当前目标

我们先需要一个可信的自有Transformer算子基线：实例级替换进LingBot-VA真实模型后，与Torch2.9的输出、有效KV/状态和RNG一致，性能接近。在此基础上才研究投机多候选调度/MegaTransformer，不能为了论文创新绕过正确性。允许直接复用对应Torch/FlashAttention开源源码，保留版本、许可证与来源；不能把调用原生SDPA冒充自有CUDA实现。

## 先读证据与代码

1. `.pipeline/experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md`、`SUMMARY-COMBINED.json`、`PRECISION-GROUPED.csv`、`FULL30-IO.csv`、`PERFORMANCE-ONLY.csv`、`PERFORMANCE-COVERAGE-CAUTIONS.json`。
2. `.pipeline/experiments/operator-torch-alignment-20261018/PLAN.md`、`SOURCE-COMPARISON.json`、`BUILD-ONLY-CHANGE.patch`。并阅读`RESULT.md`、`RESULT.json`、`ATTENTION.csv`、`BLOCK.csv`、`FULL30.csv`、`POLICY.csv`、`CPU-SAVED-OUTPUT-AUDIT.json`，以本提交真实终态为准。
3. `.pipeline/cycles/wam-research-20261016/PROGRESS.md`与`DISCUSSION-SUMMARY.md`。`review/RESULTS.json`保留旧paused72科学身份，不是本次最新算子结果。
4. 旧独立实现：`wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile/{setup.py,wam_transformer_tile/__init__.py,csrc/standalone_flash_sdpa.cu,csrc/vendor/flash_attn/softmax.h,csrc/vendor/flash_attn/flash_fwd_launch_template.h,csrc/vendor/flash_attn/UPSTREAM.json}`。
5. 新独立版本：`投机采样/experiments/operator_torch_alignment_v2/{setup.py,csrc/standalone_flash_sdpa.cu,integration.py,saved_input_validate.py,worker_full.py}`。
6. 实际模型：`投机采样/wam-p0-worktree/wan_va/modules/model.py`中的WanAttention.forward与WanTransformerBlock.forward，以及`wan_va/wan_va_server.py`实际CFG/动作处理。
7. 数值通过的对照：`投机采样/experiments/shared_prefix_torch_v2/{setup.py,csrc/shared_torch.cu,csrc/shared_flash_fwd_kernel.h}`；候选实现`投机采样/experiments/wam_candidate_tile_v1/`；真实验证逻辑`投机采样/experiments/operator_kv_matrix_v1/{worker_validation_v7.py,saved_input_io_v9.py}`。

若文件、原始tensor或关键编译证据缺失，请列具体路径和UNKNOWN，不根据名称猜测。

## 固定环境与真实输入

A100-SXM4-40GB，SM80/108SM。Torch2.9.0+cu126，git `0fabc3ba44823f257e70ce397d989c8de5e362c1`。FlashAttention `979702c87a8713a8e0a5e9fee122b90d2ef13be5`，CUTLASS `e51efbfe18fe4f4cbb66ab814c55bf4aa0185491`。本地nvcc12.4，wheel CUDA12.6；源码一致不代表SASS一致。

BF16、H24、head_dim128。原始B2 video Q128/K272、1136、2144；原始B2 action Q16/K288、1152、2160；verifier B14 action Q16/K288、1152、2160。B14是7候选×CFG2，候选分别Q16，不能当成一个可跨候选注意的Q112序列；两CFG共享组不同，各候选private KV不同。

54点覆盖三KV×三输入类型×first/last×层0/8/29；完整30层验证仅6个真实B14冻结query，未证明完整机器人闭环。原始tensor、模型权重及.so不公开，GitHub只提供源码与结构化结果；不得声称已独立重跑未公开tensor。

## 已观察的负结果和正结果

旧standalone在54点单attention/block通过旧`abs(diff)<=.02+.02*abs(ref)`，多数并非逐位。B2完整block最大绝对误差.25，不能把该容差简称统一2%。完整30层同时替self/cross：6/6非逐位、3/6甚至不满足旧容差；有效action CFG最大绝对误差.1875、rel_L2最大约.03512。

shared-prefix/guarded仅替self、cross仍Torch，在6个完整B14 query上raw/CFG输出、输入字节/全部有效KV/RNG逐位通过。G1只有kernel证据，不能借用其完整模型证书。这些有限正结果不证明任意输入，也不覆盖接受路径/scheduler/完整chunk/机器人任务。

本次已执行的**单因素修正**：旧standalone含`--use_fast_math`，通过的shared编译没有它；新v2仅移除该数学编译选项，其他变化是扩展名/命名空间隔离。源码和vendor逐字比较未改变内核数学。新结果：108attention/54block、六完整30层及三短中长完整canonical policy配对均逐位/有效状态通过；CPU27保存输出检查相符。旧同九输入仅3/9逐位，新9/9。支持该编译因素在这些真实输入上的因果影响；尚未定位FTZ/除法等具体微观指令，不能推成任意输入或闭环证书。**新二进制性能已补测**：阅读`performance/RESULT.md`与全部CSV。self/cross Graph-core中位比1.00144/1.00193，block .98465但9/54慢>5%、最大1.34367；六full30中位.99393、最大1.06654，两例慢>5%，各仅3样本。15项保存输入/原始输出CPU同源全等，其他未保存中间张量不声称独立重算。空载NVML24–25%异常仍在，不签所有形状稳定不退化，不能借旧二进制时间或最好点宣布收益。

旧新性能54点/684单元，单进程3轮；standalone self/cross Graph-core相对Torch比例中位数约.9812/.9817，但完整block14/54较慢，比例范围.2465–3.0713，有未定位波动。NVML空context非零计数未解，现有延迟属于探索性；不能把最好点或单attention均值作为完整模型稳定对齐。

## 请给出的具体答案

A. 沿实际调用链解释：Torch参考究竟进入什么后端/dispatch/split/tile，独立实现哪里相同、哪里不同。逐条引用仓库路径和行号，区分OBSERVED/HYPOTHESIS/UNKNOWN。
B. 审查移除fast_math的受控证据是否充分、剩余边界是什么；参考固定Torch源码与官方CUDA文档，检查FTZ、除法/平方根、FMA/UNFUSE_FMA、scale、softmax、split/combine及编译版本。不要仅凭选项名断言根因，也不要把未查到的全局Torch编译选项当成不存在。
C. 若局部一致而完整30层仍不一致，给出有优先级的最小定位流程：self-only/cross-only、逐层首次分叉、输入布局/stride、权重与GEMM、CFG后处理、有效KV/RNG。每项给观察信号、反证、预算与停止条件；不要建议无边界重编Torch或大规模架构搜索。
D. 给出可执行的替换验收矩阵：B2 video/action与B14 action，短/中/长；kernel→完整block→30层→完整policy chunk。明确逐位、数值容差、任务等价三种标准及各自允许主张。为什么BF16小误差可能在后段放大，需要哪些实测才能证明？
E. 审查性能计时，区分prepared kernel/Graph-core、含layout/pack的block、30层、policy/E2E；建议公平的同输入交错顺序、进程重复和误差条。给出如何在后续无争用窗口辨别~1–3%差异与异常长尾，而不排除慢点。
F. 若直接复用Torch源码，列出最小必需文件/构建边界与许可证处理。以实际固定版本为准，不能建议调用Torch SDPA再宣称独立实现；也不能把既有shared-prefix/普通缓存包装为新MegaTransformer贡献。
G. 最后给一条默认行动路径：先做什么、何种结果才继续、何时停止，以及达到“可靠替换基线”后如何开展B14多候选调度。请输出claim—evidence—gap表与1–2天优先级计划，避免再给宽泛研究大纲。

## 保留的项目边界

原FAILED/旧阈值/retention/旧实验预算不改。四项`engineering_quality_screen`、`independent_test_complete`、`noninferiority_established`、`formal_performance_permission`均false，不能因算子通过而置true。TF闭环结果与本次算子基线分开。共享GPU只做精度，正式性能待无争用。禁止新大融合、新训练/TEST、第二环境、PF3/4搜索；不把减少target calls写成E2E加速。

优先引用固定提交的PyTorch/FlashAttention源码、NVIDIA官方文档和原论文，精确说明来源支持什么；不要为迎合论文叙述跳过负结果。
