# 单层Transformer优化证据索引

更新：2026-10-23。供论文方法、消融与限制分析追溯；这是研究记录，不是论文初稿。当前为主体阶段同步、FFN四组依赖流水版，含按需调制、RMS+RoPE、Cross K/V预计算及FFN写回融合，baseline/投机生产未修改。后续沿此索引维护，原始报告仍为对应实验的权威证据。

## 最新：局部warp准备与LN调制融合（10/23下午）

主线FFN2保持四warp MMA，四其余warp准备7KiB首8行片段；B/C有限门通过、诊断看到重叠，但正常C/A变化在约±0.1%，净收益INCONCLUSIVE、不采用。B额外barrier控制不纯、诊断local改变与两次基础设施/审计失败均保留。不能声称warp专职流水已获得完整层收益。

副线L1合格后扩L13，原Welford及BF16顺序保持。最终两进程A/L1/L13/D/G同轮L13延迟0.866534/1.036672/1.049011ms，较A降0.687%/0.572%/0.428%；相对原Torch D1.3029/1.8530/2.9994×，对强G短快1.61%、中长慢3.60%/3.39%。有限225调用/候选通过，阶段耗时本轮未测，不能挪旧值。grid23→21、数学tile2736/2808/2736、local1040B，仅隔离研究采用，原使用A保留。

固定源码1ef5b83af38d523b835f2511018334ba99a4b694；16:26 owned清理，保守累计21237.866/36000秒。四false、生产未部署。原tile/投机/机器人队列不改。[回传](../local-warp-pipeline-20261023/REVIEW_RETURN.md)、[全表](../local-warp-pipeline-20261023/STAGE-TABLE.md)、[资源](../local-warp-pipeline-20261023/RESOURCE-AND-STOP-REPORT.md)。

## 最新：六投影8warp与QKV联合队列（10/23）

其余三投影复用已测8warp函数，完整层改善约1.5%–2.0%；更大M128 tile有限通过但慢9%–11%，K128默认库布局编译拒绝0GPU，均保留。恢复64×128后，独立Q/K/V各96任务共享288领取队列，grid27→23；对六投影版仅0.08%–0.44%小幅信号。最终四臂同轮对用户上版延迟降2.211%/1.995%/1.841%，原Torch完整单层1.2967/1.8469/2.9910×；不跨轮乘收益、不外推E2E。

三候选各225次有限门通过，新版全阶段重测，QKV联合比值约0.86×；不是完整异步ready调度。两机采用，算子owned清理；GPU0临时交给RoboTwin到约10点，后继部分和私有暂存仅建议未实现。

[结果](small-projection-followup-20261023/RESULT.md)、[新版全表](small-projection-followup-20261023/STAGE-TABLE.md)、[身份](small-projection-followup-20261023/SELECTED.json)、[资源](small-projection-followup-20261023/RESOURCE-AND-STOP-REPORT.json)。

## 最新：独立8warp Self QKV（10/23）

固定CTA64×128、96tile/投影、K64与两段BF16提交。共用函数混合4/8warp使整层慢1.72%–2.02%；保留原4warp函数、独立8warp QKV后同轮快1.769%/1.819%/1.438%，两机采用。两版各225次有限门通过。新版完整层原Torch比值1.2661/1.8074/2.9322×；当前所有阶段已重测，QKV仍未追平Torch。编译栈帧208→混合256→独立8B不等于动态HBM归因。没有扩大tile、没有恢复全层ready。

[结果](self-qkv-warp-control-20261023/RESULT.md)、[新版完整阶段表](self-qkv-warp-control-20261023/STAGE-TABLE.md)、[身份](self-qkv-warp-control-20261023/SELECTED.json)、[资源](self-qkv-warp-control-20261023/RESOURCE-AND-STOP-REPORT.json)。

## 最新：外层FFN内联与尾K64（10/23）

已采用两项顺序单因素：外层流水内联当轮整层延迟降0.42%/0.54%/0.61%；12个M32尾块K32→K64再降0.79%/0.42%/0.56%。每项9真实/36结构、225次有限逐位/状态通过。当前完整层原Torch比值1.2426/1.7780/2.8859×；不乘旧收益、不说E2E。static local1016B，不等于动态local消失；shared112384B、108CTA不变。

当前阶段表已补齐计算/tile/warp/同步及Torch诊断比值，两进程改良插桩资源同、整层扰动≤0.462%。FFN合计约1.01/1.01/1.03×，六小投影约0.68–0.74×；自有核内阶段与Torch热阶段缓存上下文不同，不把差值当可直接省的时间。首次插桩与两个编译上下文改变的独立FFN重放保留，精确生产独立FFN比值仍UNKNOWN。

[结果](ffn-pipeline-inline-20261022/RESULT.md)、[当前阶段表](ffn-pipeline-inline-20261022/STAGE-TABLE.md)、[身份](ffn-pipeline-inline-20261022/SELECTED.json)、[资源](ffn-pipeline-inline-20261022/RESOURCE-AND-STOP-REPORT.json)。远端权限限制已确认，后续不再试NCU/提权/驱动修改。

## 已核对的改动与结果

|改动|动机与实现|已观察结果|状态与证据|
|---|---|---|---|
|Cross K/V写回布局与向量化|FP32 shared accumulator行stride128→136，四元素搬运，保留原MMA与BF16边界|独立投影NCU shared写bank conflict从每投影2,064,384降至0；当轮完整层短/中/长延迟降3.772%/3.422%/3.147%|已采用；[原报告](projection-gemm-core-20261021/RESULT.md)。这是写回实现优化，不是新调度算法；该报告的tile/版本为历史当轮配置|
|夜间组合优化，最终候选11|轻量阶段静态步进；嵌入已有guarded self-attention并保留非16尾G1；每CTA两组4warp attention；FFN1 M128×N256×K32；去掉冗余入口同步|当轮完整层由2.510/2.718/2.977ms至2.296/2.453/2.476ms，约降8.52%/9.75%/16.81%；仍慢同轮最强guarded约10.94%/12.17%/12.50%|作为后续底座；[原报告](overnight-alignment-20261022/RESULT.md)、[身份](overnight-alignment-20261022/SELECTED.json)。上述为组合收益，不能分摊给每一项|
|依赖就绪tile队列|用领取/完成发布替代多数阶段grid同步；39次grid barrier降至1|完整层短/中/长3.765/3.887/3.858ms，相对当轮阶段版慢64.22%/58.20%/55.72%；有限精度通过|已回退，保留负结果；[原报告](ready-tiles-20261022/RESULT.md)。CTA累计领取/广播周期约21%，不能等同墙钟占比；不是普遍否定就绪调度|
|按需调制融合|将temb+固定表的FP32加法移入M1/M3与R1/OUT消费位置，保留舍入和其余阶段同步；普通路径不物化MOD|少2016逻辑tile、一次grid同步、15.75MiB中间张量；完整层延迟下降1.005%/0.979%/0.942%|当前采用；[原报告](modulation-ondemand-20261022/RESULT.md)、[源码差异](modulation-ondemand-20261022/CHANGE.patch)、[版本](modulation-ondemand-20261022/SOURCE-MANIFEST.json)|

前期逐元素/归约与投影对齐可按需追溯：[轻量算子](minor-ops-20261021/RESULT.md)、[小投影设备函数](small-projection-function-20261021/RESULT.md)、[FFN1](ffn1-tail-control-20261021/RESULT.md)。这里不补造未重新核对的逐项收益；完整实现尝试及失败保留在各原目录。

## 最新完成：FFN四组依赖流水（小幅正向信号）

四组分别等待28个FFN1生产tile，再派24个FFN2半段；第二段MMA可并行，仅提交等待第一段。9签名/36结构有限逐位与状态通过。两进程相对原Torch完整层1.1646/1.6808/2.7386×，比同轮此前阶段版快0.342%/0.357%/0.332%，幅度小，不外推应用收益。debug确认提前消费，但最后生产与完成均为组3，关键串行链保留。warm数学tile不变、grid29→27；发布/轮询开销未做NCU归因。

R1额外串行化负结果和R2产物目录导致的0调用FAILED保留。R1二进制已从原对象恢复且完整身份一致；R2在新retry目录同截止验证。两机研究入口采用，四科学权限false。

[结果](ffn-four-group-pipeline-20261022/RESULT.md)、[R2源码](ffn-four-group-pipeline-20261022/r2/CANDIDATE-R2.cu)、[计时](ffn-four-group-pipeline-20261022/WHOLE-LAYER-TIMING.csv)、[资源与清理](ffn-four-group-pipeline-20261022/RESOURCE-AND-STOP-REPORT.json)。

## 上一项：FFN写回融合（未提速）

FFN1+GELU、FFN2+残差3已实现并保留原BF16舍入。9签名/36结构有限逐位及状态通过；同轮相对原Torch为1.1646/1.6893/2.7445×，比自身融合前慢1.09%/1.14%/0.83%。按用户融合方向保留为后续流水的研究入口，不宣称性能改进。warm3400/3472/3400任务、29grid；无新的warp重叠，旧源码/二进制保留。当前仅与原Torch比较，历史guarded证据不删。

[结果](ffn-epilogue-fusion-20261022/RESULT.md)、[改动](ffn-epilogue-fusion-20261022/CHANGE.patch)、[两进程计时](ffn-epilogue-fusion-20261022/WHOLE-LAYER-TIMING.csv)。首CPU编译遮蔽错误已修复并保留，所有owned清理；累计8948.557/36000秒。

## 上一项：Cross K/V预计算

自有与强guarded均缓存最终K（含原RMS）及V。自有warm短/中/长0.957/1.124/1.140ms，仍慢强guarded7.94%/11.70%/11.69%；自身较分组重算快17.7%–20.6%。同条件未缓存时差异约±2%，缓存前后需分别报告，不把去除共同重复工作算新调度创新。

[完整结果](cross-kv-precompute-20261022/RESULT.md)、[逐阶段与扰动](cross-kv-precompute-20261022/STAGE-TABLE.md)、[失败与资源](cross-kv-precompute-20261022/RESOURCE-AND-STOP-REPORT.md)。9签名/27结构/三KV缓存baseline有限逐位与状态通过；profile-r2输出/状态通过，首诊断漏计同步的FAILED保留。warm5304/5376tile、31grid。原baseline/投机/原tile未修改。

## 上一项：Cross K/V按条件分组复用

每层保留两组条件，B14候选只重复Q及私有状态；K/V与K RMS分别1344/1344/3584→192/192/512 tile。9签名81次、27结构108次有限逐位与状态通过。完整层短/中/长2.258/2.424/2.442→1.205/1.372/1.388ms，降46.63%/43.38%/43.17%。相同分组重算guarded为1.226/1.345/1.360ms，预计算投影guarded为0.958/1.078/1.091ms；尚未超过最强缓存路径。

[结果](cross-kv-reuse-20261022/RESULT.md)、[差异](cross-kv-reuse-20261022/CHANGE.patch)、[原始计时](cross-kv-reuse-20261022/WHOLE-LAYER-TIMING.csv)。DSOcc2656e7；普通6200/6272tile、36grid同步。去重是共同工程优化，不冒充候选调度新贡献。用户现已批准双方读取预计算最终K/V，并补逐阶段计时；新阶段已收口，见上方预计算结果。

## 上一项单因素对照：RMS+RoPE融合

B14 self Q/K在同一行归约任务内旋转，保留BF16中间舍入和原FP64计算。9签名/21结构有限逐位与状态通过；短/中/长整层同轮2.270464/2.434816/2.449152ms→2.261504/2.424576/2.438400ms，延迟降0.395%/0.421%/0.439%，两进程均改善。仍慢最强guarded9.32%–10.93%。少336逻辑tile、两处grid同步和2.625MiB中间张量；没有新的动态HBM或Tensor Core利用率结论。

[结果及身份](rms-rope-fusion-20261022/RESULT.md)、[差异](rms-rope-fusion-20261022/CHANGE.patch)、[计时](rms-rope-fusion-20261022/WHOLE-LAYER-TIMING.csv)、[Tensor Core核验](rms-rope-fusion-20261022/TENSOR-CORE-AND-ROPE.md)。采用此有限版本；原调制结果以下保留为历史单因素对照，不把两轮比例相乘。

## 上一项单因素对照：按需调制

|历史KV|修改前ms|修改后ms|延迟下降|同轮最强guarded ms|当前仍慢|
|---|---:|---:|---:|---:|---:|
|288|2.292224|2.269184|1.005%|2.070528|9.59%|
|1152|2.458880|2.434816|0.979%|2.186752|11.34%|
|2160|2.471936|2.448640|0.942%|2.200576|11.27%|

测量范围：远端A100-SXM4-40GB，B14/Q16完整单层，uncached context，普通debug=false CUDA Graph，不含Python分配。两个独立进程，各臂预热600次、24轮交错采样；表为两进程中位数的均值。准入及case边界无外部GPU进程。原始逐进程数据：[CSV](modulation-ondemand-20261022/WHOLE-LAYER-TIMING.csv)。不是30层、完整机器人任务或E2E收益。

精度范围：9真实签名81次、21结构变体84次完整层调用；输出与23共有中间量逐位相同，max_abs=0，有效KV、输入/RNG、普通/debug及合法cache转移通过。有限fixture不证明全部在线输入。[9签名](modulation-ondemand-20261022/SIMPLE-GATE-RESULT.json)、[21结构](modulation-ondemand-20261022/STRUCTURAL-RESULT.json)。

源码身份：whole_layer.cu `9edf85a2226d31864e8e77bfdd3f2a419f1bf8c95bf562d4d8ae4661170d0051`；DSO `41af9a655a2387e4871eb5ab5dee1a9ac0cf613fd35c66f1f076fc5ef0f20b41`；已发布源码快照 `6bbf32b804845e9abd1e30aa85dd2d5ab11b7920`。原始身份见[本机部署](modulation-ondemand-20261022/DEPLOYMENT-LOCAL.json)。

## 论文可用表述与缺口

最新可行性核查：[Cross K/V分组复用](CROSS-KV-REUSE-AUDIT-20261022.md)、[online RMS调研](ONLINE-RMS-REVIEW-20261022.md)。初次CPU/源码审计为历史；后续分组复用已有限验证，见上方新结果。跨调用预计算已有限验证，见最新结果；已有条件缓存不算新调度贡献。

- 可以表述：本次按需调制在限定真实输入和执行边界下，去除中间表物化并降低约1%完整层延迟；历史就绪队列在当前细粒度任务组织下出现明显退化。
- 不能据此表述：峰值显存或HBM实测下降15.75MiB、整个模型/任务加速约1%、所有输入逐位等价、已超过最强B14基线，或已证明投机与新调度协同。
- 后续需要同策略/同工作量强基线、候选规模与执行引擎交互、合法提交成本及完整任务证据；现有四项科学权限仍false。
- 分轮延迟下降不可相乘作为论文最终加速比；最终表必须用冻结版本同轮重测。
- 已实施调制按需融合和self RMS+RoPE；FFN两处写回融合和跨调用预计算已有限验证；新ready或warp重叠调度未实施。

每次后续更新至少保留：问题与改动、单因素或组合身份、实际源码/二进制、同轮基线与输入、速度和精度、适用边界、资源、负结果/回退与原始证据链接。原始失败与协议不改写。

## 2026-10-22 FFN诊断完成（未实施新优化）

独立FFN原Torch比值0.8080/0.8095/0.8102×；整层嵌入未见显著新增退化。最后四生产tile使组3延后；末尾计算/写回占主导，提交等待很小。NCU诊断路径及生产SASS表明设备函数保存恢复带来local访问，未做消除收益实验。当前R2源码/DSO不改，三类测量/权限失败保留，GPU清理。[诊断与边界](ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)。

## FFN1尾部行块分摊（2026-10-22晚间）

仅尾4个96有效行块拆为12个M32任务，112→120生产任务，FFN2保持。有限9真实/36结构通过；原Torch完整层1.1910/1.7241/2.7993×，本次同轮R2回归增量延迟降1.932%/1.918%/1.856%。debug最后组相对首组就绪间隔缩短约29us；末端消费反而略增，static local1144→1224B，未重测动态计数。两机研究采用，不称独立FFN已对齐或E2E收益。累计11295.646/36000秒，owned清理。[结果](ffn-four-group-pipeline-20261022/tail-row32/RESULT.md)、[阶段表](ffn-four-group-pipeline-20261022/tail-row32/STAGE-MAP.md)。

## FFN2写回与生产函数内联（2026-10-22晚间）

A四元素写回/列常量外提，完整层同轮延迟降1.611%/1.349%/1.334%；B仅内联FFN1生产函数，同轮再降1.020%/1.156%/0.811%。当前原Torch完整层1.2185/1.7605/2.8473×；独立FFN0.9182/0.9167/0.9161×，仍慢约9%。两项各9真实/36结构有限逐位/状态通过；副本测量扰动≤0.24%。static local1232→920B，动态流量未测；不把各轮百分比相加、不宣称FFN已对齐。两机研究采用，owned清理，累计12338.592/36000秒。[结果](ffn-epilogue-abi-20261022/RESULT.md)、[阶段表](ffn-epilogue-abi-20261022/STAGE-MAP.md)。

## 2026-10-22T22:19:55.013997+08:00 当前FFN诊断（无优化变更）

B不改；末组FFN1就绪207–209μs、FFN2尾部计算92/写回18/等待1μs；local29.8MB，额外DRAM/整数指令。三KV两进程有限门通过。不能把NCU barrier42.5%当可省墙钟。[结果](ffn-current-profile-20261022/RESULT.md)。
