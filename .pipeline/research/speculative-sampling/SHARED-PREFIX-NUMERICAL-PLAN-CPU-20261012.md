# 投机采样：共享 prefix attention 数值归因与修复方案（仅 CPU）

2026-10-12，Asia/Shanghai。用户仅授权制定方案、CPU 分析和 HW 记录，**没有授权实现新 kernel、启动 GPU、训练或改变验收门**。本轮不创建 GPU job、不排队，不改公共模型或已执行的 v3 kernel/worker，不开启旧 TILE/M04 的 Cycle。

## 1. 本步完成情况

- 原模型、任务包装、本地 Torch 2.9.0+cu126 / Triton 3.5.0 源码及六份已存在的 sm80 编译产物已核对，共 30 份静态源码/产物证据。
- 完成 **13 个 CPU 代数/隔离检查**及 **3 个种子的概率精度 × 分块研究**；未导入生产模型或 kernel，未编译 Triton，未执行 target，`CUDA_initialized=false`。不是新增 42 项生产回归，更不是 GPU 修复通过。
- 1544 个已有来源/失败证据/记录的 SHA 保持不变；新报告和研究脚本独立保存。旧 v3 仍为正式 **FAIL_STRICT_BITWISE**，无新的性能结论。
- 只读子代理独立核对源码、布局和编译行为；主代理执行 CPU 研究、整合和记录。用户不需要控制子代理。

[本步结果](../../../review/OMISSIONS.md)、[静态对照](../../../review/OMISSIONS.md)、[CPU 小矩阵研究](../../../review/OMISSIONS.md)、[既有真实 GPU 失败](SHARED-PREFIX-CFG2-NUMERICAL-RESULT-20261012.md)。

## 2. 固定合同与当前证据缺口

固定合同：CFG 历史 **两路**，每路 **七候选共享同一历史**，batch14；原 `K/V_pool[2,2160,24,128]` 零拷贝，有效槽 2144；Q/currentK/currentV `[14,16,24,128]`，原候选顺序 `[0×7,1×7]`。prompt 不共享；不改 Q/K norm、RoPE、cross attention、投影、scheduler、阈值、bonus、selector 或 noise=0 写 KV 规则。

原路径先按 mask 的物理槽升序 gather history2，再按 rowmap 选为 history14，拼各自 current16。新路径以 slots/rowmap 虚拟读取同样的联合序列：

`O[b,h] = softmax(Q[b,h] · [K_history[rowmap[b],slots,h]; K_current[b,h]]ᵀ / sqrt(128)) · [V_history; V_current[b,h]]`

原 SDPA 将 BSHD 转 BHSD；新 kernel 直接按 BSHD strides 读取，**不是漏做 head 转置**。没有跨候选 attention，也没有把历史和当前各做 softmax 再相加。长历史最后一块为 **32 history + 16 current + 16 masked**，实数数学和静态索引意图一致。

仍缺少：原 A100 **实际 SDPA backend**、真实 attention 输出 stride、全部真实 query 的原始 QKV 以及归因时完整 prefix/slot 内容对照。旧逐层数据仅覆盖第一条 query，未比较 S/R currentV，S 末尾 state/RNG、GPU 失效/隔离未完成。

因此当前只能把“执行路径变化和层间传播”视为有力解释，不能认定无 bug，更不能保证某个 CPU 改法一定逐位通过。

## 3. 原 SDPA 与新 kernel 的差异表

| 项目 | 原路径：确认与未知 | 新 S：已确认 | 归因动作 |
|---|---|---|---|
| backend | `custom_sdpa → F.scaled_dot_product_attention` 已确认；内部 flash/efficient/cuDNN/math 选择 **UNKNOWN** | 固定任务私有 Triton | 先记录真实分派，不将 `attn_mode='torch'` 等同 MATH，不以 CPU backend 推断 A100 |
| 参数/数学 | 默认 mask=None、dropout=0、causal=False、GQA=False、scale=1/√128 | 无 dropout/causal；联合 softmax；scale×log2(e) 对应 exp2 | 先核对单位、可见 token 和槽序，不能盲目加 causal 或删 log2(e) |
| QK | BF16 输入；具体内部算术/分块未知 | BF16 MMA，FP32 累加 | 已有 PTX 是 `bf16×bf16→f32`；metadata 默认 tf32 **不是本次用了 TF32 的证据** |
| softmax | 不同 backend 可能在线/融合；不能假设都是 eager exp | BN64 在线 max/denom/acc；`ex2.approx.ftz.f32` | 区分算法正确性、指数实现和归约次序 |
| 概率/PV | 实际 backend 的概率舍入点未知；本地文档说明 MATH 对 BF16 输入使用 FP32 中间量 | P 先转 BF16，V 为 BF16，PV 累加 FP32；denom 使用未转 BF16 的概率 | 优先核查概率舍入及它与在线重缩放/tile 的相互作用 |
| scale/FMA | scale 位置、是否融合减 max 未记录 | 已有 PTX 出现 scale-minus-max 的 `fma.rn.f32`，编译 `enable_fp_fusion=true` | 固定 logits/精度，单独对照融合；不同时改 exponent/tile |
| 分块/归约 | 原 backend 的 tile、warp reduce、MMA 累加次序未知 | BM16/BN64，4 warps/2 stages；2160 tokens 共34块 | 先确定原路径，再选择一个 tile 对照；不做无预算搜索 |
| 最后除法 | 原算法可能采用不同归一化实现，未知 | 已有 PTX 是 `div.full.f32`，不是已观察到 rcp.approx 或 div.rn | 作为低优先级对照，不把猜测写成已确认差异 |
| 输出布局 | R attention 的 stride 未留档 | `torch.empty(q.shape)` 为 contiguous BSHD | 记录 R stride；必要时用仅输出 `empty_strided` 对齐，保持历史不展开；排除下游 flatten/GEMM 路径改变 |
| 输出 dtype | BF16 | 最后舍入 BF16 | 更接近 FP64 不等于匹配原 BF16 字节；正式比较不改变 dtype、不忽略 signed zero |

依据：[原模型 model.py](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py)（37–40、608–660、687–690）、[新 kernel](../../../投机采样/experiments/shared_prefix_gpu_v3/prefix_kernel.py)、[本地 SDPA 文档](../../../review/OMISSIONS.md)（5805–5893）、[分派声明](../../../review/OMISSIONS.md)、[已存在的匹配长历史/原池 stride 编译产物](../../../review/OMISSIONS.md)。cache 证明编译行为，不是完整 launch trace。

## 4. 本轮 CPU 小矩阵研究结果（不是 GPU 仿真）

输入保留两路历史、14 候选、current16、head_dim128；为控制研究规模只用 **1 head**，不使用 checkpoint/model。独立 FP64 联合 softmax 对照在线 exp2 的数学，历史长度 0/1/16/63/64/65/127/128/129/2144，14 候选均通过；最大绝对差 **1.4433e-15**。当前候选单行扰动后，其他 13 行 CPU oracle 输出不变。

精度/tile 研究固定相同 FP32 logits、相同 V 值与 CPU GEMM，仅改变 P 的 BF16 舍入和 BN64/128，三个种子均保留所有样本。相对 RMS 对 FP64 数学输出（最终 BF16 舍入前）：

| CPU 研究变化 | 三种子实测相对 RMS 范围 | 可推出什么 |
|---|---:|---|
| P32、BN64 对 FP64 | 2.716e-7–2.751e-7 | 该 synthetic CPU 设置中 FP32 路径数学误差小 |
| P 先转 BF16、BN64 对 FP64 | 0.15349%–0.15686% | P 舍入可以产生可见局部误差，是优先排查点 |
| PBF16 下 BN64 与 BN128 相比 | 0.04680%–0.04766% | 在线块边界/重缩放与概率舍入存在相互作用 |
| P32 下 BN64 与 BN128 相比 | 2.631e-7–2.683e-7 | 该设置中单纯 FP32 归约差异远小于 PBF16 差异 |

独立 scale-minus-max 舍入例中，模拟单次/两次 FP32 舍入的 **62/256** 个标量不同，最大差 **4.7684e-7**。这只说明融合可改变字节，不能确定原 SDPA 的融合策略或误差占比。

**CPU BLAS/exp2/reduce 不是 Triton BF16 MMA/SFU/FMA 仿真；1 head synthetic 不是实际 24 head/29 query。** 数据只能支持排查优先级，不能将 0.154% 当实际 GPU 误差分摊，也不能说取消 PBF16 就“修好了”。原融合 backend 也可能舍入概率；提高数学精度反而可能更远离它的逐位输出。

## 5. 五类假设与验证顺序

| 假设 | 当前判断 | 如何区分/反证 |
|---|---|---|
| H1 新旧 backend 算术/归约路径不同 | 最优先；v3 算术已确认，原 backend 未知 | 固定真实 R backend 身份、同 QKV/可见 prefix 与重复 R；不能偷换为 forced MATH gold |
| H2 PBF16 舍入与块内 max/重缩放相互作用 | CPU 有力支持，真实 GPU 占比未知 | 固定 logits 与 PV 算术的 CPU 单因素；真实输入分别量化数学误差与 R 字节匹配 |
| H3 scale/FMA、exp2、归约/除法顺序 | 新产物确有这些算术；旧方细节未知 | 一次仅改变一个因子并保留 baseline、负结果、编译选项；不一口气改所有数值路径 |
| H4 slots/CFG/currentV/prefix 状态错误 | 静态未见明显错位，不能排除真实输入相关 bug | 先比较真实槽序、rowmap、Q/currentK/**currentV**、有效 KV 内容和全层 state/RNG，NaN 无效槽不读取 |
| H5 输出 stride 改变下游投影执行 | 已确认 S contiguous；R 未知 | 首先采集真实 R stride；输出 layout-only 对照，在值/算术不变时检查投影与完整模型 |

P0 是身份和数据可比较性关，优先于任何数值改法。通过后按 **layout-only（若确有差异）→概率舍入诊断→scale/FMA→指数实现→tile/归约→QK/PV 更高精度诊断** 的顺序，依据真实 backend 修正排序。仅最有证据的一项进入候选修复，不做组合爆炸或自动重跑。

## 6. 任务专用诊断/候选修复设计（尚未实现）

### 6.1 第一阶段：把真实参考身份和证据补齐

默认关闭、仅实例级采集，在原 first candidate 的原始调用上记录：Torch/build/库配置、实际 ATen backend 事件、原 Q/K/V 的 dtype/shape/stride/offset、R attention 输出 stride、mask/causal/dropout/scale/GQA、有效 slots/rowmap、模型/缓存代次和版本。仅后续获 GPU 授权时记录实际 CUDA backend；CPU 只能测试记录器，不能冒充其值。

保存第一条 query 的 **layer0、layer8、layer29**：真实 prefix K/V 有效值（CFG2），Q/currentK/currentV、R attention 输出及 slots/mask/id/is_pred。用 CPU contiguous 独立拷贝保存值，另存原 GPU stride 元数据，防止序列化一个小 view 时隐含保存整块 history14 storage。

每层 BF16 数据预算为 prefix K/V 50.25MiB + Q/K/V/out 5.25MiB = **55.5MiB**；三层约166.5MiB，拟设总导出上限 **192MiB**（含 metadata/文件开销），超限停止，不自行追加多 GiB fixture。全层有效 history/metadata/RNG 仍在 RAM 中比较，未分配槽不序列化。

包装必须返回原调用的原始输出，不把诊断结果送入 scheduler/selector；实例 backend、forward、全局 flags 在 finally 中恢复。采集代码不驻留在未来 OFF 计时段，不改冻结 golden，不重新定义原参考。

### 6.2 第二阶段：单因素矩阵与最小修复

| 对照 | 固定什么 | 改什么 | 用途/限制 |
|---|---|---|---|
| O：输出布局 | 原 S 数学/索引/CFG2 | 仅匹配真实 R 的输出 stride | 若有真实 layout 差异先排除，原 prefix 仍零拷贝 |
| P：概率精度研究 | 固定 logits、V、归约设定 | 是否对 P 舍入 BF16 | CPU 可纯单因素；GPU 高精度实现不是“删一行 cast” |
| F：融合 | 同 scale/base/dtype/tile | fp fusion on/off，记录实际 PTX | 禁止同时改指数与分块；关闭融合不保证匹配原 backend |
| E：指数 | 同 log2 logits/scale/tile/类型 | 单独对照 exp2 的实现 | `tl.exp` 未必是独立高精度路径；明确 libdevice 版本和编译代码 |
| T：分块 | 同 dtype/base/fusion | 一个选定 BN/归约策略 | CPU P 舍入交互已提示不可只看整体误差；不搜索一堆参数 |
| A：高精度诊断 | 原 BF16 值及数学合同 | QK 或 PV 的运算精度，分开试 | 数学 oracle 接近不等于 R bitwise；不得把它直接当部署修复 |

本地 Triton 明确要求普通 `tl.dot` 的两个输入类型匹配：**P32 × VBF16 不合法**。若做 FP32 PV 诊断，须把 V 的 BF16 值精确提升为 FP32并显式 `input_precision='ieee'`，否则默认 TF32 引入新因素。该分支改变算术实现，只用作高精度诊断，不能宣称 GPU 上纯粹隔离了 P 舍入，也不保证性能。

最终方向是 **保留共享访存，尽量复刻已确认原 backend 的算术和布局**，而不是盲目“更高精度”。若原 backend 的计算拓扑/舍入无法在现共享实现中匹配，应保留失败结论并停下，不改阈值、不强行部署，也不能保证一定存在高性能的逐位修复。

### 6.3 正式验收门保持不变

- GPU 数学诊断与正式 PASS 分开；CPU 近似误差、FP64 更接近、既有 1% corruption ceiling 都不是正式容差。
- 候选只有同实际默认 R 的 BF16 **字节逐位一致**才可进入后续：30 层、29 变化原始 query、全部输出/状态/RNG，十二类失效、其他13行隔离。不可改 R 为 forced backend 来制造 PASS。
- state/RNG 检查应设计为即使出现数值不一致也在安全清理路径记录检查结果和异常；它是诊断证据，不把 FAIL 改成 PASS，不继续性能关。无法安全检查则明确 UNKNOWN。
- 之后才做六组平衡完整 verifier OFF/显存，冷租约/索引/搬运/guard/日志/释放计费。仍不能代替闭环成功率、环境步数与 E2E。

## 7. 后续 GPU 提案与预算约束（未授权、未启动）

建议先完成并 CPU 验证上述默认关闭观测包装；**不急着实现一组新数值 kernel**。新 GPU 的第一目标是确认 backend 和导出真实 QKV，而非测速或训练。

可单独提案：**1 worker、52 次 target、最多24次额外直接 SDPA、15分钟安全限时、0 episode/EGL/完整chunk/环境步/训练**。原 capture 已包含52次 target，内部 attention 不另重复计作 extra direct；3个选定层各2次默认 R 重复=6，至多4个 forced backend 各3层=12，共18个额外意图，6个余额不是重试授权。forced backend 仅诊断，不替换默认参考；不支持的分支按预声明负例记录并计费，不回退伪造成功。默认 R 重复不逐位、状态异常、资源/超限等真正关卡失败即停。预计8–12分钟，15分钟含加载、观测、IO和清理；这是提案估计，不是运行承诺。

当前累计仍为 **294/900 target 意图、660.551/2700秒**，余额 **606次/2039.449秒**，本步未消耗 GPU。若单独执行52次采集，余554次，**不足再完整重跑原585次计划**；不能承诺在原总预算内完成“独立采集＋原全套成本主测”。可在以后明确设计的完整 job 内复用其原52次 capture 做观测，或在独立诊断后另确认后续预算；不得悄悄删测试/退还已耗预算或自动连跑。

本报告不创建这种 job，不自动等待空闲，不启动 GPU。任何后续实施和 GPU 预算仍单独确认。

## 8. 下一步建议与记录

**建议下一步仅 CPU 实现、验证默认关闭的原 SDPA backend/布局记录器和真实 QKV/prefix 导出包装**，补齐真实 production-loader 的接口与默认关闭不扰动、状态/flags 恢复、导出大小/独立存储、预算计数回归。原 GPU backend 在 CPU 结果中必须保留 UNKNOWN。数值 kernel 本身先不改，GPU/训练不自动启动。

包装 CPU 回归通过后，再单独确认最小 GPU 观测预算；取得实际 backend/QKV 后，CPU分析真实数据并提出一个最小算术/布局候选修复，然后单独确认其验证预算。每步继续在投机采样 HW 留下结果、未验证项和下一步。

研究脚本初稿有括号解析错误，在任何 import/执行前失败，0 GPU/模型/target；原稿与 log 留档，新 v2 AST 检查和上述 CPU 研究完成。没有覆盖旧失败或修改生产 kernel。[v1 留档](../../../review/OMISSIONS.md)、[实际研究脚本 v2](../../../review/OMISSIONS.md)。
