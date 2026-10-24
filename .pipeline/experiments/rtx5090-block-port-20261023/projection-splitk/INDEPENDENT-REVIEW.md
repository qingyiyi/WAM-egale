# 三个小投影split-K：源码与结果复核

初次源码复核：2026-10-24T01:32:29.657747+08:00。本次仅CPU只读比较候选与冻结A，未运行GPU/SSH。本文件由参与实现的agent复核，不能替代独立于实现者的审查；另一个调度审查agent的结果见`SCHEDULE-AUDIT.md`。数值/性能结果待根agent回传后补审，不提前判定采用。

## OBSERVED：源码改动范围

仅`workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`改变。冻结A为上一轮已保留FFN2 split-K版本；不是早期串行FFN2。

- 新增`projection_partition8`（候选行173）与`projection_commit8`（行199），从既有`cutlass_gemm.cuh::self_qkv8_tile`的SM120路径提取单K片段主体。
- `self_qkv_projection8`（行232）仅SM120且`wi=9/13/19`进入新路径，依次对应Self输出、Cross Q、Cross输出。每投影仍96个M64×N128输出tile，改为960个片段计算任务，`partition=task/96, tile=task%96`。
- 每投影增加97个int：有界CAS领取计数与96个提交计数；三个投影独立，共增1164B global。入口在原`begin_stage`的grid同步前清零，出口仍为原grid同步。
- normal路径无新结果审计字典；debug新增`projection_splitk_claimed:[3]`与`projection_splitk_commit:[3,96]`。期望分别全960与全10。原`p.queues[st]`不再代表该投影有效任务领取数，应使用新字段。

逐函数文本比较确认未变：`ffn2_partition64`、`ffn2_commit64`、`action_ffn1_tail`、`action_ffn1_produce`、`action_ffn2_splitk`、`action_ffn_pipeline`、`self_qkv_projection_group`、`action_projection`。删除新增SM120分支后，原`self_qkv_projection8`fallback逐字相同；`cutlass_gemm.cuh`、build脚本及Python数学实现未由本agent修改。

## OBSERVED：数学与同步边界

- 原`CutlassSelfQKV8::Mma`、8个MMA warp、M64/N128/K64、三级cp.async主循环和shared结果stride136保持。
- K分段仍九段320加尾段192，合计3072；每段同样重新清FP32 accumulator，iterator起点为`partition*320`，有效K迭代数仍5或3。没有把尾段填成320次有效计算。
- 片段输出先形成FP32部分和。若不是首段，先加前一段BF16结果转FP32；仅末段再加bias；最后按原四元素向量路径舍入BF16。没有改变舍入次数或加法顺序，没有用浮点atomic归并。
- 本CTA shared保存本片段FP32结果等待提交，完成计算与转存后均保留CTA同步；整个提交完毕前不领取下一任务、不复用shared。
- 每tile的前驱片段总先取得owner，提交只等待更小partition；cooperative grid全部驻留时不形成等待未派发前驱的环。后继通过acquire、CTA同步读取前驱结果；所有写线程fence后经CTA同步，由leader release发布完成，最后CTA同步再领取任务。
- CPU映射检查确认960对(tile,partition)唯一，且每个非首段的前驱任务编号严格更小。这是调度结构检查，不能代替GPU无死锁、计数与逐位验证。

## 初审UNKNOWN：当时待实际结果（后续状态见追加审查）

- 新正常/插桩二进制是否编译成功，实际寄存器、静态local、shared、resident CTA数是否变化，待读取构建及GPU属性。虽然`Params`和`Shared`声明未变，helper拆分及控制流变化仍可能改变编译器资源使用，不能提前宣称资源完全不变。
- 当前三个小投影每片段仅5或3次K64迭代，相比FFN2每片段32次，队列/等待/发布开销占比可能更高；这是待测假设，不是慢或快的已知原因。
- 是否有限逐位通过、三档正常完整层是否稳定改善、阶段耗时与Guarded比较均未在本文件提前推断。所有采用判断等待两个独立正常计时进程及public入口验证。
- 原Torch对应480个CTA与本候选960个片段任务不能混称相同tile配置；本候选刻意保持当前自有M64/N128以只检验K分段并行。

## 文档遗留

源顶部“小投影由warp3–6做MMA”及投影附近“两段K1536”注释属于SM80历史，不能用于解释当前SM120活动路径。当前SM120已核为8warp、十段K；本轮冻结数学源码期间仅记录过时注释，不为此额外改源码或二进制。

## 初审结论

源码范围与预定单变量组织变更一致，未发现阻止进入有限GPU精度门的数学/同步问题。尚未形成实验采用结论；四科学权限不变。

## 结果追加审查：2026-10-24T01:37:27.424640+08:00

**结论：REJECT_PERFORMANCE，保留失败候选，当前恢复A=SPLITK-M64-FFN2。** 本次只读复核结果与普通源码字节比较，未运行GPU、未修改源码。上述候选行号应从`candidate-source/csrc/whole_layer.cu`读取，当前optimized已回退，不能继续将当前路径解释为960投影任务。

### OBSERVED

- `results/precision-data/RESULT.json`为`FINITE_EXACT_PASS`。三KV所有27项捕获值、debug/normal输出、有效KV、输入、RNG通过。两计时进程的前后四臂输出亦逐位通过；三投影领取计数均960、三组96提交计数均10，FFN2提交96项仍均7。上述逐字段读取核验，不只采信status。
- 两独立进程正常完整层相对A均退化。按两进程中位数均值复算`Own/A−1`：短 **42.69994%**、中 **40.87264%**、长 **38.08657%**。分别看两个进程，短42.57279%/42.82713%，中40.72368%/41.02204%，长37.34317%/38.82930%，退化方向稳定。
- 候选阶段诊断显示三个投影分别约0.162–0.176ms，同口径独立Torch投影约0.0417ms；Torch/候选投影约0.237–0.256。三个投影诊断区间合计短/中/长0.49912/0.51456/0.522496ms。阶段区间含同步，与正常完整层计时分开，不能当作可精确相加的因果分解。
- 候选正常kernel资源为`[170,1,100096,255,784,256]`，与A记录一致：CTA数、每SM CTA、shared、寄存器、静态local及线程数没有改变。不能把本次退化归因于这些静态属性恶化。
- 当前`optimized/csrc/whole_layer.cu`与`A-source/csrc/whole_layer.cu`普通字节比较相同；候选源另保留于`candidate-source`。本项仅证明所读cu已回退，不替代二进制/public入口回退核验。

### 支持与不支持的因果结论

支持：在固定原MMA/tile/K分段/舍入方案下，本轮新增的跨CTA片段领取、等待与提交组织整体导致性能显著退化；候选诊断瓶颈集中在改动的三个小投影。FFN2的成功不能外推为短K片段同样受益。

不支持：没有GPU硬件计数器或分项消融，不能宣称完全由某一种atomic、threadfence、特定等待、HBM/L2流量或某条指令导致；255寄存器/784B静态local相同也不能证明动态spill流量完全相同。每段5/3次K64计算的计算量较FFN2更小、领取更集中，是源码可见差异，但各自延迟贡献仍UNKNOWN。

### 结果追加审查时尚待收口的证据（已由下方最终补审关闭）

根agent正在重测回退后当前版本的阶段表，并核验public默认/自定义stream；本次追加审查时这些新结果尚未读取，不提前写为通过。后续结果应单独追加，不把失败候选的阶段时间作为已保留版本的当前时间。四科学权限保持false。

## 最终补审与关闭：2026-10-24T01:40:22.522950+08:00

**终态：CLOSED_REJECTED_CANDIDATE_RETAINED_A。** 本次只读取回传结果并对本地文件作普通字节比较，未运行GPU或改源码。

- public默认/自定义stream分别ID0/192870656，实际loader加载`wam_whole_splitk_cuda`；两份结果均`FINITE_EXACT_PASS`。两者明确`expected_splitk=true`、`expected_projection_splitk=false`：保留FFN2 split-K，三个小投影恢复原实现。六case的27项捕获值、normal/reference、normal/debug、有效KV、输入、RNG及FFN2的96×7提交计数全部逐字段通过。
- `results/RETAINED-stages-data/RESULT.json`为`COMPLETE_EXACT_EAGER`，正常与插桩输出通过，当前版本三投影阶段约0.05962–0.06197ms、完整FFN区域约0.29541–0.29725ms；失败候选0.162–0.176ms的投影时间不属于当前保留版本。当前单轮Torch/Own完整层约1.15354/1.61967/2.04044，只标为本轮单进程复测，不冒充新的双进程重复。
- **A列身份特别说明：** RETAINED复测复用了上一轮脚本，其`loaded_modules.A`为`ffn-splitk/A.so`及`wam_whole_grain_cuda`，即更早的串行FFN2 GRAIN；不是本轮`projection-splitk/A.so`（已含FFN2 split-K）。该A列不能用作本轮候选与回退对照；本轮候选退化结论仍来自`PROJECTION-measure1/2`的匹配A。当前复测的Torch/Guarded/Own和阶段数据可按真实身份使用。
- `OWNED-CLEANUP.json`于01:38:10记录16个PID+出生身份均已退出、无GPU compute进程、显存0MiB、未kill；`instance_shutdown=false`，不能宣称租机实例已关闭。
- `RETURNED-BYTE-CHECK.json`六份源码/二进制回传一致。本次另直接读取本地optimized的五份源码文件与对应`A-source`以及所选二进制与`A.so`，六份普通字节比较均相同；没有计算SHA。

全部等待回传、public回退核验和owned清理状态已关闭。保留本轮负结果与候选源，实际版本为SPLITK-M64-FFN2。未测计数器、未分解调度子项的贡献、有限fixture范围及四科学权限false等限制继续有效。
