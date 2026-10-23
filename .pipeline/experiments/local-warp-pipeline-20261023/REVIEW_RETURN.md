# 局部warp流水判定回传

完成时间：2026-10-23T16:26:34.464876+08:00。锚点319ac6a17daed34e80692796f72a97bf44f4a99f；文档锚点effd6580b43b503ce6f00b0074f8cc8570adffa1。

**结论：FFN2主线未得到可信完整层净收益，不采用；副线保留LN1＋LN3调制融合L13，完整层比A快约0.4%–0.7%。中长KV仍未超过已有强guarded基线。** T0至T4全部完成；没有扩展DAG/FA2/tile搜索。仅隔离研究采用，原baseline、投机生产与两机原使用入口不改，四项科学权限及production_admitted全部false。

## 固定源码和结果入口

- 保留源码commit：`1ef5b83af38d523b835f2511018334ba99a4b694`，文件`算子研究/wam_ops/optimized/csrc/whole_layer.cu`。本地独立worktree已提交；GitHub发布状态另见PUBLICATION-RECEIPT.json，未有收据前不表示远端可读。
- 完整源与DSO身份：[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)，A独立原二进制保留；L13源码SHA256 `d714ebfe1e838d7be59e37122f0f392c92d765d4792cfb97d8f38e4260191dfb`，DSO SHA256 `164e42cb4bfe88f67277cb76f789e2c5941ce5ef0bf7b61812951a233b5304c9`。
- 本轮唯一新采用patch：[T3-L13-build/CHANGE.patch](T3-L13-build/CHANGE.patch)；负候选由make_bc.py和锚点可重建，BC普通/诊断源码均保存于variants与独立补丁。
- 完整阶段表：[STAGE-TABLE.md](STAGE-TABLE.md)；结果：[FINAL-COMPARISON.md](FINAL-COMPARISON.md)；逐样本：[WHOLE-LAYER-SAMPLES.csv](WHOLE-LAYER-SAMPLES.csv)。本报告末尾附全表。

## OBSERVED：主线A/B/C

A为独立锚点二进制；B/C同一新DSO，W3–6四warp MMA、128×128tile、K7168×2、BF16 partial、任务队列和八warp提交保留。W0/1/2/7提前准备第二段当前tile的bias/table及首8行gate/R2，7KiB放在原shared union允许空间。只覆盖满tile6.25%的行，尾tile8.33%；没有提前读取尚未发布partial。

B先准备并全CTA会合再MMA；C准备与MMA重叠再会合。**控制不完全：B的额外串行barrier也出现在无准备的第一段**，C/B收益包含这次同步消失，不是干净的重叠因果分解。A未被塞入新分支大kernel。

每臂9真实/36结构，225次完整调用有限逐位、全部有效KV、输入/RNG与合法状态通过。诊断补测42调用验证A诊断副本与A有效中间量，以及B/C首段BF16 partial、最终FFN2/output；没有把allocated未初始化尾部当有效状态。

每档288个第二段任务的软件代表warp区间：B重叠0，C重叠288，C重叠P50约3.072–4.096μs。诊断BC local1048B、普通1016B，不能把诊断171μs MMA时间当普通路径时间/NCU吞吐。只支持代表warp区间重叠，不支持全CTA关键路径或TensorCore活跃度上升。

无插桩C/A：1.000821/0.999460/1.001044，延迟变化+0.082%/−0.054%/+0.104%，小于/接近T0 A/A波动（观察范围约−0.290%至+0.102%）。三档均INCONCLUSIVE净收益，**C不采用**；未消耗额外算法修订或展开扫参。见[T2-RESULT.md](T2-RESULT.md)、[T2-DIAGNOSTIC-SUMMARY.json](T2-DIAGNOSTIC-SUMMARY.json)。

## OBSERVED：副线与最终同轮比较

**计时条件：A/L1/L13/D/G 全部开启外层 CUDA Graph，包含自有实现。** 自有算子内部不依赖 Graph，计时脚本仍捕获并重放它；不存在本轮仅 guarded 使用 Graph 的差别。双方 eager 对照本轮未测，见[源码定位与口径](MEASUREMENT-MODES.md)。

L1仅在A上合并LN1与调制；通过门且两进程三档均无可信退化后扩L13。保留原四warp Welford树、FP32加乘顺序和BF16输出；8warp从行内shared消费，普通路径省去N1/N3两个FP32全局中间量。B2仍走原路径。LN1和LN3各少336独立调制tile与一次grid，完整grid23→21，数学tile2736/2808/2736。非新的候选图调度。

L1、L13各225次有限门通过，包括B2 action、B14 action、B2 video三档KV，以及有效长度尾部、mask hole、CFG/context换序与temb不同广播形式。原23共有中间量逐位一致；Graph重放、cache与RNG通过。有限fixture不证明所有输入；B2性能本轮未测。

L13资源：108CTA×8warp，shared112384B、255寄存器/线程、local1040B（A1016B），1CTA/SM。没有动态spill计数，不把static local变化当根因。

|KV|A ms|L1 ms|L13 ms|D ms|G ms|A/L13|D/L13|G/L13|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|288|0.872525|0.868147|0.866534|1.128986|0.880717|1.0069|1.3029|1.0164|
|1152|1.042637|1.039027|1.036672|1.920973|1.000627|1.0058|1.8530|0.9652|
|2160|1.053517|1.050189|1.049011|3.146394|1.014605|1.0043|2.9994|0.9672|

T4-final1/2正常无插桩warm单层Graph，GPU0无其他compute进程，GPU1机器人继续；相同最终Cross KV预计算、相同真实query和有效历史。24交错样本/臂/档、每样本10 replay；表为两个独立进程中位数均值。D=原Torch静态dense历史，G=已有guarded；不是完整Python动态入口、30层、chunk或E2E。

L13相对A三档均有小幅正向，在T3和T4共四个进程方向一致；不称统计显著。相对G短快1.610%，中长慢3.602%/3.391%。LN3对L1的增量很小，T3中档约0.047%，不可宣称单独显著；最终同轮仍不见可信回退，所以保留同一简单融合模板。

## 推导、假设与UNKNOWN

- 源码可确认：LN融合减少N1/N3的FP32中间写读和两次grid；不能分离这两者各占多少收益，未实测HBM节省或动态spill。
- 假设：FFN2原bias/table已外提，可隐藏准备仅一小部分，有限重叠被布局/提交开销抵消；当前只是与结果一致的解释，没有逐项因果归因。
- UNKNOWN：完整在线输入/全部layer数值，B2新产物速度，30层/完整verifier/合法提交成本及机器人E2E；H800/5090D迁移性能。没有GPU counters，不补造带宽/occupancy实测（这里的1CTA/SM仅资源约束计算）。
- 本轮不足以支持“新warp流水优于强B14”或“投机与MegaTransformer已证协同”。

## 资源、失败与停止

本轮GPU占用窗口保守46.833分钟；GPU worker生命周期4.841分钟；CPU编译12.640分钟，CUDA active UNKNOWN。累计保守5.899/10小时。远端证据约209.3MiB。全部owned退出，GPU0释放；原机器人队列不变。两次失败（TMPDIR、无效尾部审计）及真实消耗保留，无timeout。详见[RESOURCE-AND-STOP-REPORT.md](RESOURCE-AND-STOP-REPORT.md)。

## 下一轮最多三个讨论项（未自动启动）

1. 中长KV仍落后强G约3.4%–3.6%，先讨论是否继续聚焦投影/FFN primitive差距，避免将此次微小LN收益包装成主要调度创新。
2. FFN2局部准备净收益未定；若继续，需先证明更多准备覆盖能在原数值/存储边界内成立，而非扩大无界搜索。也可直接止损此路线。
3. 若补跨架构，用户当前无H100；H800可作为候选，独占/计数器权限优先。SM120的99KiB/CTA低于当前109.75KiB，5090D需另做布局适配；未授权迁移或购卡。

硬件/论文核验另见[PUBLICATION-AND-HARDWARE.md](PUBLICATION-AND-HARDWARE.md)。原LingBot-VA已正式RSS2026，论文部署RTX5880Ada，但其5v/10a与我们配置不等。

## 附：最终完整阶段表

**T4 完成｜2026-10-23T16:26:33.795787+08:00**
本次评估：最终同轮A/L1/L13/D/G。当前实际保留：**L13（仅隔离研究采用，未部署生产）**。
源码 `d714ebfe1e838d7be59e37122f0f392c92d765d4792cfb97d8f38e4260191dfb`；独立二进制 `164e42cb4bfe88f67277cb76f789e2c5941ce5ef0bf7b61812951a233b5304c9`。
A100-SXM4 GPU0；B14/Q16/D3072/H24/FFN14336；KV288/1152/2160；双方最终Cross K/RMS/V预计算。108CTA×8warp；shared112384B；255寄存器/线程；静态local1040B；1CTA/SM。结构来自源码核对；阶段耗时本轮未插桩。

|ID／阶段|数学工作与形状|tile与任务数|warp分工|同步与数据交接|
|---|---|---|---|---|
|01 layout|有效历史slot压紧|按pool/256轮；CTA0|8W|CTA0搬运后grid1；slots就绪|
|02 调制参数|table+temb，FP32|消费处生成；无独立tile|消费warp|无独立屏障|
|03 LN1＋调制|一行D3072，Welford LN→同一行调制|224行tile|W3–6归约，8W搬运|原行内归约/CTA屏障；合并出口grid1|
|04 调制1|BF16[N1×(1+scale)+shift]|并入前项，无独立tile|8W vec4|并入前项；保留BF16边界|
|05 Self QKV|3×[224,3072]×W；各K1536×2|64×128，合计288tile|8W cp.async/MMA/写回|队列广播/两段提交；grid2|
|06 Q RMS+RoPE|两行D3072 RMS后复数旋转|112tile|两组4W|各组归约＋CTA；grid1|
|07 K RMS+RoPE|两行D3072 RMS后复数旋转|112tile|两组4W|各组归约＋CTA；grid1|
|08 private KV|候选suffix私有，历史只读|无插入tile|无|无；由attention分别寻址|
|09 Self attention|FA2 Q64/K64/D128；有效Q16/32|96/168/96tile；KV循环5/18/34|两组4W；QK/PV MMA|FA2组内barrier；grid1|
|10 Self输出|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|11 残差1|BF16[x+gate×SO]|2048元素×336tile|8W vec4|grid1；R1就绪|
|12 LN2|D3072 LN＋affine|224行tile|W3–6归约，8W搬运|行内barrier；grid1|
|13 Cross Q|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|14 Cross K|context×Wk，warm预计算|warm不执行；cold192tile|warm无|warm无|
|15 Cross V|context×Wv，warm预计算|warm不执行；cold192tile|warm无|warm无|
|16 Cross Q RMS|两行D3072 RMS|112tile|两组4W|组归约＋CTA；grid1|
|17 Cross K RMS|最终K已预计算|warm不执行；cold512tile|warm无|warm无|
|18 Cross attention|Q16×K512×D128；KV64×8轮|168tile|两组4W FA2|FA2组内barrier；grid1|
|19 Cross输出|[224,3072]×W；K1536×2|64×128，96tile|8W MMA/写回|队列/两段提交；grid2|
|20 残差2|BF16[R1+CO]|2048元素×336tile|8W vec4|grid1；R2就绪|
|21 LN3＋调制|D3072 LN，FP32输出→同一行调制|224行tile|W3–6归约，8W搬运|原行内归约/CTA屏障；合并出口grid1|
|22 调制3|BF16[N3×(1+scale)+shift]|并入前项，无独立tile|8W vec4|并入前项；保留BF16边界|
|23 FFN1|D3072→14336；原BF16边界|108×128×256/K32；12×32×256/K64|主8W MMA，尾4W；8W写回|组生产28/28/28/36 release；无grid|
|24 GELU|BF16[tanh-GELU(BF16 FFN1)]|并入23，无额外tile|8W CUDA core|同23；ACT发布后可读|
|25 FFN2|14336→3072；K7168×2|128×128；48输出/96分段tile|W3–6 MMA；8W提交|mainloop bar2/128；CTA；原partial acquire|
|26 残差3/FFN出口|BF16[R2+gate×BF16 F2]|并入第二段FFN2提交|8W CUDA core|partial/输出release；最终grid1|
|27 最终退出|B14无cache回写|无计算tile|108CTA全体|grid1，返回输出|

循环内同步沿原实现：LN保留Welford树及行内屏障；GEMM保持原K循环与cp.async等待；FA2保留组内barrier。没有把动态barrier次数猜成常数。

|阶段／可比区间|自有短/中/长|Torch短/中/长|Torch/自有|测量状态|
|---|---|---|---|---|
|01 layout|—|—|—|NOT_MEASURED|
|02 按需调制|—|—|—|NOT_MEASURED|
|03–04 LN1/调制|—|—|—|NOT_MEASURED|
|05 QKV|—|—|—|NOT_MEASURED|
|06 Q RMS/RoPE|—|—|—|NOT_MEASURED|
|07 K RMS/RoPE|—|—|—|NOT_MEASURED|
|08 private KV|—|—|—|NOT_MEASURED|
|09 Self attention|—|—|—|NOT_MEASURED|
|10 Self输出|—|—|—|NOT_MEASURED|
|11 残差1|—|—|—|NOT_MEASURED|
|12 LN2|—|—|—|NOT_MEASURED|
|13 Cross Q|—|—|—|NOT_MEASURED|
|14–15、17 Cross K/V/RMS|—|—|—|NOT_EXECUTED（warm预计算）|
|16 Cross Q RMS|—|—|—|NOT_MEASURED|
|18 Cross attention|—|—|—|NOT_MEASURED|
|19 Cross输出|—|—|—|NOT_MEASURED|
|20 残差2|—|—|—|NOT_MEASURED|
|21–22 LN3/调制|—|—|—|NOT_MEASURED|
|23–26 FFN共同区间|—|—|—|NOT_MEASURED|
|27 退出|—|—|—|NOT_MEASURED|

阶段值未测，不借用历史数字。Torch独立热Graph与kernel内区间的缓存/边界不同，阶段值不可相加替代完整层。

|KV|A ms|本次变体 ms|A/变体|D/变体|G/变体|波动与结论|
|---|---:|---:|---:|---:|---:|---|
|288|0.872525|0.866534|1.0069×|1.3029×|1.0164×|相对A延迟降0.687%；相对G延迟差-1.610%|
|1152|1.042637|1.036672|1.0058×|1.8530×|0.9652×|相对A延迟降0.572%；相对G延迟差+3.602%|
|2160|1.053517|1.049011|1.0043×|2.9994×|0.9672×|相对A延迟降0.428%；相对G延迟差+3.391%|

T4正常无插桩、同轮五臂、两独立进程，24交错样本/臂/档、每样本10次Graph replay，150次预热。完整block调用精度/状态通过。保留L13小幅改进，FFN2重叠C不采用。L13短KV快于G，中长仍慢约3.4–3.6%，不宣称超越全部强基线、不外推完整WAM/机器人。所有单阶段时间NOT_MEASURED。
