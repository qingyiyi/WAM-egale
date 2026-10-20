# 当前单层persistent实现：内部FA2＋双缓冲GEMM

[whole_layer.py](whole_layer.py) → [csrc/whole_layer.cu](csrc/whole_layer.cu)是当前入口。整层所有数学在一次cooperative kernel内完成；host只做参数绑定、检查与workspace分配。不调用host GEMM/SDPA或设备子kernel。**研究实现，完整block未通过原Torch逐位门，production_admitted=false，不默认安装到模型。**

## 源码地图

| 文件 | 作用 |
|---|---|
| `csrc/whole_layer.cu` | 整层DAG、队列、Norm/RoPE/调制/残差、KV、资源与launch |
| `csrc/gemm_pipeline.cuh` | 64×64×32 WMMA GEMM，SM80 cp.async双缓冲、LD40 |
| `csrc/fa2_stage.cuh` | self/cross FA2任务、split选择、同kernel combine |
| `csrc/fa2_load.cuh` | FA2的逻辑KV→共享prefix/私有suffix地址适配 |
| `fa2_codegen.py` | 从冻结vendor提取原FA2设备函数，适配worker/同步/加载 |
| `build_whole_layer.py` | 构建独立`wam_whole_fa2_cuda`，不会加载旧whole DSO |

上游FA2/CUTLASS及baseline冻结依赖均只读。源码来源和生成文件hash随[构建证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/fa2-evidence/generated/FA2-SOURCE.json)保存。

## DAG、tile与线程

调制表→Norm1/调制→self QKV→QK RMSNorm/RoPE→合法KV→self FA2→输出投影/门控残差→Norm2→cross QKV/QK RMSNorm→cross FA2→投影/残差→Norm3/调制→FFN1/GELU→FFN2/门控残差。

当前108个常驻CTA，每CTA256线程/8warp，仍按阶段grid barrier保证依赖；任务粒度为CTA，不是每warp独立领取整个tile。

|阶段|warp分工|tile|
|---|---|---|
| GEMM | 0领任务、1/2加载A/W、3–6 WMMA、7写回bias/BF16 | 输出64×64、K32；每计算warp32×32 |
| FA2 | 0领任务；3–6执行原FA2 load/MMA/softmax/store；其余等待 | 普通Q128/KV64；split Q64/KV128；head128 |
| Norm | 1加载、3–6按原Welford树归约、7写回 | 完整3072维行 |
| pointwise | 原阶段分工 | 128元素块 |

短Q16使用FA2合法尾块，B14是14个独立实例，不是Q112序列。公共history和私有suffix地址直接读取，没有复制14份history或跨candidate互相注意；没有新增跨候选共享tile贡献。split-KV合并使用原softmax统计合并算法并仍在同一kernel内。named barrier2限attention的128参与线程，barrier1限norm组，CTA和grid barrier保持。

FA2加载仍在其四个计算warp中，尚未做额外producer/consumer warp分离；也没有跨阶段ready队列或跨CTA普通shared memory。运行时255registers/thread、82464B shared、2168B localSizeBytes、理论1CTA/SM；实测occupancy、spill流量未知。

## 数值与实测

当前真实范围B2/B14 action Q16、B2 video Q128，D3072/H24/d128/FFN14336、cross512、短中长KV。BF16输入输出、FP32调制/Norm参数、原FP64 RoPE、GEMM/GELU舍入边界保留。不是任意shape/训练API。

[最新完整报告](../../../.pipeline/experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)：18相同Q/K/V的self/cross对照逐位PASS；9重复/输入/RNG/参数/有效KV和18写转移通过。完整block仍9逐位FAIL，relative L2约0.133%–0.563%，最大hidden绝对误差0.5–1.5。Q、FFN2有同输入独立差异；video短/中Norm2仅12/8个元素差异；长video直到GELU逐位一致、首次分歧在FFN2。

两进程216未插桩样本：较旧async，短action慢8%–9%，长action耗时降18%–21%，长video降43%；仍为native的3.20–7.60倍耗时。18trace各一个compute kernel。不是chunk/30层/机器人E2E，也不是最强cached/Graph对照。

## 构建与调用

使用固定Torch2.9提交与SM80私有CUDA工具链；设置`WAM_FA2_BUILD_ROOT`为可写构建目录，再从本目录执行`python build_whole_layer.py build_ext --build-temp ... --inplace`。在`torch.inference_mode()`内构造`WholeLayerBlock(block)`，debug返回中间量、metadata及仅初始化有效前缀的logical_slots。生产入口不改。

旧M34、M37、M38和原M39双缓冲测量、失败/预算审计均保存在对应HW报告。旧PipelineBlock只用于已固定M39对照DSO；历史生成器需要原SHA源码，不用于当前FA2构建。下一步先对齐FFN2/Q/Norm2数值，再处理资源压力与细粒度依赖调度。
