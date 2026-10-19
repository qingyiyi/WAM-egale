# WAM完整Transformer block：协作接口与tile研究交接

本文件可以直接转给协作者。当前可复现代码固定为 WAM-egale commit 99f7c2f3e845920fd83b22fedd19e72089f3a886；后续优化使用同一算子研究目录的optimized入口，baseline作为只读对照。本文核对结构和形状，不宣称新的tile/persistent优化已实现或已获得收益。

## 1. 优化对象

对象是冻结LingBot-VA/WAM大模型的一个完整Wan/DiT风格block，共30层；不是53070参数的TF64 draft。block为带时间调制的PreNorm self-attention、text cross-attention和GELU FFN，self/FFN含gate残差。D=3072，24头，每头128，FFN宽度14336，LayerNorm/RMSNorm eps=1e-6。

当前baseline/optimized均有完整本地C++/CUDA数学链。GEMM依赖cuBLASLt，Norm/Flash使用固定上游源码；Torch负责Tensor/allocator/stream等。当前host顺序调用多个kernel，不是已完成的persistent block。cuBLASLt的host调用不能直接搬进CUDA while循环，persistent内需要可设备执行的tile原语。

## 2. 实际API与数据类型

```python
Y = block(X, context, temb, rotary_emb,
          update_cache=0, cache_name='pos', candidate_cache_rows=None)
```

| 输入/输出 | 普通action/CFG2 | 七候选verifier/CFG2 | dtype |
|---|---|---|---|
| X / Y | [2,16,3072] | [14,16,3072] | BF16 |
| context，已投影的text | [2,512,3072] | [14,512,3072] | BF16 |
| temb，已投影的六组调制 | [2,16,6,3072] | [14,16,6,3072] | BF16；内部加table后FP32 |
| rotary_emb | [2,16,1,64] | [14,16,1,64] | complex64；当前旋转算术用double |
| K/V cache物理pool，各一份 | [2,2160,24,128] | [2,2160,24,128] | BF16 |
| candidate_cache_rows | None | [14]，七个0再七个1 | int64 |

cache另有id[2160] int64、mask/is_pred[2160] bool。KV和metadata是有状态接口的一部分；Y同形不代表缓存语义已经正确。update_cache=0/1/2与原写入/预测/驱逐协议保持一致，candidate路径仅允许read0。read0下只要求原声明有效历史和metadata不污染，不把原无效pool字节提升成新契约。

原text embedding是4096维，在进入block前已经投影成3072，不能用4096代替block的cross-attention输入宽度。原机器人action latent的112维也不是Transformer隐藏维度；输入嵌入和最终action/video head在本block边界外。

## 3. 短/中/长的真实长度

以下action K指attention真正可见的历史＋当前私有16 token，不是pool容量：

| action分组 | 历史L_h | 当前私有Q | self-attention有效K |
|---|---:|---:|---:|
| short | 272 | 16 | 288 |
| middle | 1136 | 16 | 1152 |
| long | 2144 | 16 | 2160 |

七候选共享各自CFG组的历史；当前K/V为[14,16,24,128]、分别私有。当前动态实现会打包成[B,K,24,128]再调Flash；已有shared-prefix强路径可避免这种历史复制，后续必须对照它。

视频参考输入是X/Y=[2,128,3072]、context=[2,512,3072]、temb=[2,128,6,3072]、rotary=[2,128,1,64]。本次真实video fixture的short/middle/long有效K分别为272/1136/2144，前置历史144/1008/2016；这些与action长度不同。pool容量同为2160，不得把视频也写成288/1152/2160。

B14=7候选×CFG2。每个逻辑attention实例的Q仍为16，不能把七候选拼成互相可见的Q112。按同CFG合法历史分组query tile是执行组织，不能改变各行可见的private KV、sigma、位置和mask。

## 4. 一次block计算

```text
X
 ├─ FP32 LayerNorm → time scale/shift → BF16
 │    → self Q/K/V投影 → Q/K RMSNorm → RoPE
 │    → 带历史/私有KV的self-attention → 输出投影
 └─ gated residual → H1
      → affine LayerNorm → text cross-attention → 输出投影
      → residual → H2
      → FP32 LayerNorm → time scale/shift → BF16
      → Linear 3072→14336 → BF16 → GELU → BF16
      → Linear 14336→3072 → gated residual → Y
```

六组调制依次为self的shift/scale/gate和FFN的shift/scale/gate；各层有自己的[1,6,3072] learned table。cross-attention有独立Q/K RMSNorm，无self的RoPE，也不增加同一套gate。

Q/K RMSNorm是对每token的完整3072维进行归约，然后才拆成24×128；不能改成各头128维独立归一化。LayerNorm同样需要完整行归约。权重主项BF16；scale_shift_table、norm2 weight/bias按原加载保留FP32。保留原BF16舍入边界，不能因为融合而自动延后舍入。

## 5. GEMM与attention的具体形状

记M=B×Q，PyTorch权重存储为W[N,K]，数学为X[M,K]@W.T。当前action普通M32，七候选M224；线性层已经能够展平这些行。

| 运算 | 每block次数 | 普通action M,N,K | B14 action M,N,K |
|---|---:|---|---|
| self Q/K/V和输出；cross Q和输出 | 6 | 32,3072,3072 | 224,3072,3072 |
| uncached cross K/V | 2 | 1024,3072,3072 | 7168,3072,3072 |
| FFN1 | 1 | 32,14336,3072 | 224,14336,3072 |
| FFN2 | 1 | 32,3072,14336 | 224,3072,14336 |

shared/cached conditioning投影属于共同强基线，不能将省掉uncached cross K/V重新申报成candidate-aware调度收益。已缓存路径需要按真实相同text/层/权重/dtype/device和生命周期复用，CFG组不能未经核验合并。

self Q/K/V逻辑布局[B,Q或K,24,128]。按单个batch/head看，QK=[16,128]×[128,K]→[16,K]，PV=[16,K]×[K,128]→[16,128]；cross对应K=512。Flash没有必须物化完整score矩阵的接口要求。B×24是逻辑batch-head实例数，不是CTA数量或满载证明。

已存在的FFN1+GELU tile原语为CTA(64,128,64)、warp(32,64,64)、MMA(16,8,16)、3级pipeline。几何推导：M32时112个输出tile、M方向有效比例50%；M224时448个tile、M有效比例87.5%。这只是该固定tile的形状利用率，不是Torch选择的实际tile、实测SM占用或已证明的整体加速。3候选/M96是后续诊断形状，现有优化原语入口仅准入M32/224，不能声称M96已验。

## 6. 真正限制tile流水的依赖

- Q/K投影某一列tile完成，不代表RMSNorm可开始；其行归约要覆盖全部3072维。
- self-attention需要当前候选全部合法私有K/V就绪；不同候选可以独立推进，不能读取别人的private KV。
- FFN1的某个N tile经GELU后，可以供FFN2对应K片段使用；FFN2最终输出必须完成14336维归约。允许研究按K片段就绪提前累积，但要保持归约/舍入语义，不能把局部和当最终结果。
- producer写workspace后再发布ready；consumer完成后才能复用buffer。跨CTA不能靠普通shared memory传递数据。
- persistent worker不能全部占着资源等待尚未执行的producer。必须给出warmup、ready依赖、在途任务上界和终止条件；ring复用另需generation，防读旧数据。
- 先在一个子图和单block内实现；跨30层会改变数据就绪和权重复用，另行证明。

## 7. MegaMoE源码如何参考

已固定官方DeepGEMM commit 1fc9c66975ee322d628b3288534ca0e9ce12a64d，只下载源码阅读，未安装/运行。scheduler/mega_moe.cuh:83定义Linear1/2任务，92定义TaskInfo，143定义按K块的上游就绪依赖，344/372动态领取并交错任务，15有warmup防死锁。sm100_bf16_mega_moe.cuh:626/687/726/816/826分别可读到activation load、weight load、MMA、scheduler和epilogue分工。

可迁移的是tile描述、常驻worker分工、producer/consumer流水、就绪与buffer生命周期；本项目映射为candidate/CFG/m/n/k tile。官方这条BF16实现使用SM100 TMA/UMMA/TMEM、双CTA cluster和寄存器重分配；A100/SM80应重新实现支持的copy/MMA/barrier，不能照搬硬件原语。MoE的SwiGLU/专家路由/NVLink也不进入本项目。源码逐项参考与SM80适配见 [MEGAMOE-NOTES.md](MEGAMOE-NOTES.md)。

## 8. 协作入口与验证边界

- 完整block：/rjs/cxz/论文探索/算子研究/wam_ops/baseline/csrc/block.cu:262
- 原生绑定：/rjs/cxz/论文探索/算子研究/wam_ops/baseline/native.py:25
- 优化入口：/rjs/cxz/论文探索/算子研究/wam_ops/optimized/block.py:6
- FFN tile原语：/rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/ffn.cu:17
- 固定Flash入口：/rjs/cxz/论文探索/算子研究/wam_ops/optimized/csrc/attention.cu
- 模型安装/还原：/rjs/cxz/论文探索/算子研究/wam_ops/install.py:49
- 精度与完整状态：tools/verify_native.py、verify_native_states.py、replay.py；计时：tools/time_native.py。

只维护baseline/optimized两个活跃实现；不改投机源、原tile或旧FAILED。已有证据是M28有限真实输入逐位与动态block计时，不是persistent实现、最强Graph增量或机器人E2E。FFN1+GELU device原语已过有限逐位门，旧device FFN2失败仍是硬缺口，当前FFN2靠匹配原Torch的cuBLASLt通过。

机器可读的九个真实签名与27参数为SHAPES.json/PARAMETERS.json；原文件路径在签名中，原件留服务器。签名来源是已序列化、归一成contiguous的逻辑CPU fixture，不声称还原生产CUDA原stride/alias/storage offset。GitHub不包含原权重/完整tensor，外部协作者未拿到原件时只能做对应范围的分析，不能声称重新验证全部真实输入。
