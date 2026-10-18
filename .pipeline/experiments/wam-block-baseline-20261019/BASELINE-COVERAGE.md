# 替换范围与验证范围

| 实际运算 | R0 | 已有R1 | 新B0默认 | 设备tile状态 |
|---|---|---|---|---|
| self SDPA | TORCH_ORIGINAL | OWN_STANDALONE | TORCH_ORIGINAL，可另挂aligned但本轮GPU未运行 | 现有独立attention kernel，不等于完整DAG内可调用原语 |
| cross SDPA | TORCH_ORIGINAL | OWN_STANDALONE | TORCH_ORIGINAL，可另挂aligned但本轮GPU未运行 | 同上 |
| self/cross QKV和输出投影 | TORCH_ORIGINAL | TORCH_ORIGINAL | TORCH_ORIGINAL，由新Python阶段组织 | GEMM仅host调用，没有新D0 |
| Q/K RMSNorm、FP64 complex RoPE | TORCH_ORIGINAL | TORCH_ORIGINAL | TORCH_ORIGINAL运算/边界 | 无新设备原语 |
| norm、调制、gate、残差、cast | TORCH_ORIGINAL | TORCH_ORIGINAL | 原Torch数学，MIXED控制流 | 无新设备原语 |
| FFN1、GELU(tanh)、FFN2 | TORCH_ORIGINAL | TORCH_ORIGINAL | 原Torch模块/functional，显式边界 | 无新设备GEMM链 |
| mask/index KV和metadata | TORCH_ORIGINAL | TORCH_ORIGINAL | 原helper与原分支组织 | 有效语义与写状态单独核对 |
| shared/guarded候选历史 | 原B14 dense materialization | 既有独立实验 | 独立adapter准备，GPU NOT_RUN | 原kernel有有限旧证据，联合新基线不能借证书 |
| context投影缓存 | 原core未缓存 | 原core未缓存 | 两个真实fixture静态CPU控制 | 普通缓存，非MegaTransformer新贡献 |

| 实现 | kernel/子层 | 完整block | 完整30层 | 完整policy | 机器人闭环 |
|---|---|---|---|---|---|
| 已有R1 v2 | 108有限GPU attention；九保存输入 | 54有限GPU点，逐位/有效状态 | 六冻结query有限通过 | 三短中长冻结配对通过 | 本轮未验证；四false |
| 新B0_TORCH | 两fixture×28可观察阶段CPU | 六真实action输入+六合法写CPU控制通过 | NOT_RUN | NOT_RUN | NOT_RUN |
| B0_ALIGNED | 接口已准备 | GPU NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 新联合R_strong适配 | 旧shared/guarded范围可查；本轮native未导入 | 联合GPU门未运行 | NOT_RUN | NOT_RUN | NOT_RUN |
| D0/P_seq/P_sched | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_APPLICABLE | NOT_APPLICABLE |

已有R1替换范围只为SDPA；验证到完整policy不扩大其实现所有权。新B0是完整顺序host基线，不能叫全自有CUDA block。collection_complete/finite/bitwise/tolerance/state/performance分字段见ACCURACY-RESULT.json。

支持证书：layer0/first、B2或CFG2×7=B14、action Q16、K288/1152/2160、hidden3072/H24/D128/FFN14336、BF16参数/主激活、指定FP32/FP64边界、original cache。三份video保存文件没有block_args与完整cache metadata，完整video B0为BLOCKED_MISSING_FIXTURE；旧video attention证据仍保留。不是随机替代或重新采集。
