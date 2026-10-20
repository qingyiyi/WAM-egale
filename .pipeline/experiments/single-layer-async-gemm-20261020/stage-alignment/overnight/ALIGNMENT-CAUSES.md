# 先对齐阶段：已确认原因与待证部分

2026-10-21夜间工作中记录；不是最终性能结论。当前按用户要求停止新增跨阶段调度，后续改动只对齐阶段实现。完整层计时为真实B14输入、uncached/eager、准备排除，非机器人E2E或最强缓存/Graph结果。

## 已观察

- 基础阶段三项后的[矩阵](STAGE-POST-RESULT.md)：cross K/V独立约1.09倍native；FFN1约1.83倍、FFN2约1.38倍，cross attention约2.5倍。这些是同输入独立阶段对照，足以排除“全部差距只来自整层同步”的解释。后续RMS/FFN/连续K/V改动之后，旧矩阵不可当作最新版矩阵。
- 当前对照原生FA2的真实kernel记录：128线程、255 registers/thread、65536B shared、grid `[1,14,24]`。见`../../context128-profile/native-middle-2.kernels.json`中的两个flash forward。文件中的estimated occupancy不是NCU测得occupancy，不引用为动态计数。
- 整层运行时查询：108 CTA、1 resident CTA/SM、256线程、98848B shared、255 registers/thread。见`rms_pair/TIMING-1.json`的resources；后续编译另核对。FA2只使用其中4个计算warp。
- 原生FFN1记录的kernel为`ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn`，grid `[56,2,1]`，256线程、238寄存器、147456B shared。结合真实M224/N14336，库tile覆盖N256/M128。自有原语此前是M64/N128，随后只把M改为128，尚未复现原库8warp/N256的组织。
- 原Norm独立kernel静态约22/38寄存器、24B shared；整层内Norm仍受整层最大资源配置约束。把每CTA一行改为两行保留了128线程行内归约顺序，九签名有限逐位通过，并使整层改善约2%。
- 地址适配的可消除开销已被单因素实验支持：cross attention改回连续K/V的原FA2加载器后有限逐位通过，完整层改善约1%；不能据此解释全部退化。

## 资源推导（不是动态利用率测量）

NVIDIA官方[Ampere调优指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)给出SM80每SM 65536个32位寄存器、164KiB shared，最多64 warp。根据上述launch参数，整层256×255=65280寄存器，一个CTA已几乎耗尽寄存器文件；第二个CTA无法驻留，这与实际CUDA occupancy查询的1一致。原FA2每CTA128×255=32640寄存器，两个CTA的寄存器和shared需求可容纳。

因此同一336个batch/head任务，在理想均匀负载下，当前108个常驻CTA需要最多4轮，原FA2允许216个resident CTA时需要2轮。它解释了并发容量的差别；不是“运行时恰好快两倍”的定理，也不是已测SM利用率。真实队列、访存、尾部和编译代码仍要计入。

## 尚不能定量归因

- 编译日志报告device函数spill，最终cubin/runtime又报告stack/local帧；编译阶段、device link结果和动态流量不同。不能把1616字节帧直接当作每tile HBM读写量，也不能在无计数器条件下断言spill占总时长百分比。
- 全局barrier和领取队列有成本，但A/B/C比较含不同编译资源、cache历史和插桩，不能简单相减当作纯同步时间。
- 短KV出现约2.12ms与2.63ms的native跨进程变化。当前保留全部交错轮次，不选快轮、不改时钟。需要时钟记录进一步解释波动；未观测前仅称状态波动，不能确定是热限频。

## 对齐顺序

保持数值门不变，依次确认当前阶段tile/搬运/归约是否接近原库，然后再讨论统一kernel对资源的约束。只在有新测量支持时保留改动；仍不能对齐则报告剩余绝对耗时与证据缺口。尚不把通用batch/cache或已有FA2称为新调度贡献。

## 后续有限结果与已排除解释

- B14 Q16改用M64 query tile，九签名有限逐位通过，整层短/中/长改善约4.8%/6.7%/9.5%；没有合并候选query，也没有改变阶段依赖。
- N256/8warp FFN1的整层负结果已在两个进程复现约3%–4%退化，shared从98848B增至148000B；它不是当前活动版本。是否主要由于shared/L1配置或编译变化尚无单因素因果测量，不能据此下最终根因结论。
- 频率日志实际观察到SM从1095MHz升到1410MHz（部分采样1320MHz）；此前“短KV状态波动”的解释获得直接频率证据。未修改GPU时钟。仍按同轮native/before/new比较，不挑快轮。
- 源码复核确认全部旧整层和阶段kernel已经使用`const __grid_constant__ Params`。因此取消原拟新增该标注的重复试验（0 GPU/0 forward），**不能把缺少该标注作为当前退化原因**。编译stack仍需从设备调用/寄存器生存期等进一步核对。

## 10/21后续受控因素（01:40之后）

- point8192：只扩大逐元素任务，九签名有限精度通过；两个完整层交错进程均慢约0.8%–1.0%，已撤回。不能假设任务更少就更快。
- gemm_inline：只强制内联GEMM设备分派。有限精度通过，但中/长仅约0.1%/0.4%、短两进程方向相反；最终stack/local1464→1640，未保留。不能把out-of-line函数存在直接等同于主要瓶颈。
- rms_vector：cross-K RMS以原四元素顺序直接向量读写，省去BF16→FP32 shared中转。有限精度通过，完整层稳定改善约1.1%–1.7%，保留。
- compact B14：4warp、2级CUTLASS和B14专用FA2空间；数学输出及有效状态有限逐位通过。资源从108CTA/256线程/98848B/255reg/1464local变为216CTA/128线程/66080B/255reg/624local。两个进程完整层却慢约2%–10%，不是入选优化。分阶段解释仍在测量，不能仅凭静态资源变化宣称原因已证明。

compact首次CPU构建因上游Config(2)选择MmaPipelined而不是cp.async multistage失败；修正为取Config(3)迭代器/策略、实际Mma(2)后完成。随后三次计时入口失败均保留：资源metadata全相等断言、FA子空间与整层workspace大小混淆、脚本文本替换产生tuple key。它们均发生在正式样本前，不能作为数学输出不一致，也不能删掉预算消耗。最终r5同源计时完成。

## 插桩与时钟的进一步边界

stage_final的A/B是CUDA Graph marker间隔，C是在另一个插桩整层中的阶段间隔；C与B还可能处于不同GPU时钟状态，现有表未做频率校正。不能把C−B全部归给barrier或寄存器。当前计划对同一resident线程同时记录globaltimer和clock64，只辅助判断阶段间平均SM频率，不把周期数解释为有效执行指令、活跃率或NCU stall。

NVIDIA原文：[`clock64()`按每个SM计周期，但包含线程未执行指令期间](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#clock-and-clock64)。独立marker若落在不同SM不能直接做cycle差；本诊断只对同一kernel内同一resident线程取差。仍不改GPU时钟。
