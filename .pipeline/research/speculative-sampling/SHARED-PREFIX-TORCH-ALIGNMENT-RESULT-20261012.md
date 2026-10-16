# 共享历史KV：直接对齐原始torch的结果

> 2026-10-12T20:01:24+08:00 | 投机采样 | zh-CN | Asia/Shanghai

共享KV已经直接对齐原始torch SDPA：复用当前PyTorch2.9对应FlashAttention计算、仅改六处KV读取地址，并移除实验编译中的fast-math。真实3层2,064,384元素、完整30层首个候选Q/K/V与attention输出、全部29个最终候选输出均逐位一致；12类缓存失效、R/HC/S的其他13行隔离、历史/metadata/RNG和原29输出hash检查通过。新S数值门PASS，性能与真实任务闭环尚未验收。

## 结论与范围

用户要求继续精度对齐，并明确要求与原始torch实现比较。本轮金标准始终是模型原调用的`torch.nn.functional.scaled_dot_product_attention`，默认实际dispatch为FlashAttention；没有替换金标准、强制MATH或放宽容差。原R的29个输出哈希与历史冻结结果相同。

**新实现的S数值验收已通过。整体性能验收、完整episode成功率及E2E改善仍待验证。** 原Triton S失败及第一版独立编译的一个元素差异均保留。

## 原因与实际修复

共享只改变历史K/V的读取地址，同一CFG下7个候选共用原两行历史池，当前Q/K/V仍各自保留14行。它不需要改变attention数学或浮点计算顺序。原问题出在优化同时把torch FlashAttention替换成不同的Triton浮点执行流程：遍历顺序、归约安排、缩放/softmax和舍入流程不同，因此相同BF16输入不保证相同输出位。

已直接阅读本机与安装torch commit对应的源码：PyTorch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`，FlashAttention commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`，CUTLASS commit `e51efbfe18fe4f4cbb66ab814c55bf4aa0185491`。来源只读；其文件哈希和许可证复制到独立投机采样实验目录，没有改变TILE研究代码或其结果。

原SM80/D128非causal路径为M128/N64、4warps、Q不驻留寄存器，倒序遍历KV；softmax分母先在线程内逐块累加，最后做四线程归约；PyTorch定义`UNFUSE_FMA`禁止一处缩放/减最大值的FMA融合。新实现保留原FlashAttention/CUTLASS计算及同步代码，仅替换6处K/V全局内存读取为“CFG2有效历史slot或该候选自己的当前token”。[逐行差异](../../../review/OMISSIONS.md)和生成器反向重建证明算术/同步源码字节未改。

第一版沿用了实验算子的`--use_fast_math`选项。三层中第0、29层已经完全一致，第8层688,128元素只剩1个不同；同版dense与共享地址输出哈希完全相同，排除了共享寻址带来的附加误差。第二版只移除该编译选项，所有三层输出全部对齐原torch。这个对照定位了第一版剩余一个元素差异的编译选项原因。

原torch SM80二进制反汇编与新版本均无`FMUL.FTZ`，并含倒数精度校正中的`FFMA.RM/FFMA.RP`；第一版使用FTZ和近似倒数路径。它支持归一化末端舍入差异的解释。单因素实验定位到fast-math选项；该选项同时影响多个浮点指令，未声称仅某一条倒数指令独自造成这一元素差异，也未声称nvcc12.4产物和CUDA12.6 wheel具有相同SASS。

## 同一真实输入的原torch逐位对照

使用前一轮已导出的第0、8、29层真实Q/K/V，不重新加载完整模型。每层Q/current为`[14,16,24,128]`、有效prefix为`[2,2144,24,128]`；同时构造原容量2160的两行池，无效槽填NaN，保持原slots和CFG rows。原torch参考可以物化14行，S路径直接读取prefix2/current14，没有历史14行或fullKV14物化。

每版24个直接attention写前意图：每层原torch、原torch重复、原Flash math dense、共享紧凑CFG2、共享原slot池、共享重复、扰动row0当前V后的原torch与共享输出。所有结果与原torch及冻结原输出比较。

| 层 | 原Flash计算独立编译，fast-math开启 | 共享地址，fast-math开启 | 共享地址，fast-math关闭 |
|---|---|---|---|
| 0 | 0个不同 | 0个不同 | 0个不同 |
| 8 | 1个不同 | 1个不同 | 0个不同 |
| 29 | 0个不同 | 0个不同 | 0个不同 |


第二版三层共2,064,384个元素，最大绝对差0、相对RMS差0；NaN无效槽屏蔽、repeat、扰动后的原torch匹配及其他13行隔离均通过。正式标准是逐位比较，绝非“误差足够小”。

## 完整模型数值门

重新加载冻结的真实模型及`T-init3-chunk14.pt`、原A_off、固定selector/threshold/seed，原cache、text encoder CPU、VAE CUDA、FSDP关闭；没有修改公共model/server/config或权重。第一段capture仍完全调用原torch，52个目标前向得到原29个candidate输入及输出。随后在同一独立GPU进程执行：

- 原HC完整29输出保持逐位一致，保留原history14基线。
- 新S完整29输出全部逐位一致，CPU独立复查了194,880个最终输出元素。
- 首个候选的全30层Q、当前K/V和attention输出与原torch各自逐位一致，排除后续传播放大。
- 每段历史及metadata、CPU/CUDA RNG保持不变，lease结束释放引用和索引。
- 12类失效全部拒绝旧lease，并将重建结果与变更后原torch比较：update_cache=1/2、clear_pred、clear_cache重建、history恢复、历史值修改、有效槽替换、容量填满后的真实eviction、普通调用、rows修改、新request/cache generation及model generation变化。
- R、HC、S各自扰动候选row0，另外13行保持逐位相同，扰动行输出有实际变化；最后R/S收尾再次通过。

本轮数值计划独立声明179个target意图上限，保留所有数值、状态、失效和隔离门。实际161次意图、159次主机返回，另2次在失效负例中按设计抛出`LeaseError`并计入预算；旧lease在model入口或层内拒绝，未消耗预留的全部失效上限。**未运行旧计划中的六组OFF性能测试，未将它们标为PASS。**

## 验证、预算与资源

两版直接对比各5项CPU测试通过，模型集成6项通过（总16项）；CPU原production loader加载30层小随机checkpoint，验证default-off、HC字节、S零拷贝与输入变化、slot mutation以及持久计数/阶段限额，CPU没有初始化CUDA。训练权重只在授权的GPU模型worker中加载。

GPU三次新独立job：第一版精确保留FAIL_NUMERICAL（7.306614秒）、第二版微测试PASS（7.686421秒）、完整模型PASS_STRICT_NUMERICAL（468.705091秒）。每个job仅启动一次worker，无失败job复用。直接attention共48次，目标模型前向161次，新增GPU含加载/IO/同步/清理共483.698125秒。

累计507/900 target意图、1256.949035/2700 GPU秒；尚余393次及1443.050965秒。所有journal闭合并独立恢复；三个自有GPU进程已退出，模型端口释放。审核2262个既有冻结文件和8个模型集成文件，state/rules/cycle保持原状态；原公共源码、TILE来源和旧失败结果保留。

本轮0完整episode、0 EGL、0完整policy chunk、0训练。包含完整verifier计算的数值验证没有设计成性能测量；不能从这些阶段的wall时间外推加速，更不能外推task成功率。共享存储也不自动代表CTA之间已有片上KV复用。

## 下一步

在相同原torch参考及逐位门下，制定下一轮有界完整verifier性能方案，比较R/S的时间、额外显存和稳定性；采用原先六组交叉顺序，不因数值通过就宣称加速。当前完整性能计划尚未执行，下一计划须携带累计预算，不能把旧585-forward计划直接重跑或默默删门。性能有收益后再考虑task0固定条件的闭环。

当前数值不一致的阻塞已经解除；下一阶段需要回答“这份共享地址实现是否值得用”，即完整verifier的时间/显存收益及稳定性。新实现是默认关闭、只在此次私有实例安装的实验算子，尚未接入公共生产路径。

## 复查入口

- [完整模型结果及清理](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/FINAL_RESULT.json)、[原29哈希](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/results/capture.json)、[全29输出](../../../review/OMISSIONS.md)、[全30 attention](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/numerical/ATTENTION-ATTRIBUTION.json)。
- [第二版直接原torch对照](../../../投机采样/results/runtime/p0-shared-prefix-torch-v2-20261012-194704/FINAL_RESULT.json)、[第一版一个元素差异](../../../投机采样/results/runtime/p0-shared-prefix-torch-v1-20261012-193802/FINAL_RESULT.json)、[第二版源码/许可证](../../../投机采样/experiments/shared_prefix_torch_v2/SOURCE-PROVENANCE.json)、[共享取数实现](../../../投机采样/experiments/shared_prefix_torch_v2/csrc/shared_load.h)。
- [CPU独立字节/计数/来源审计](../../../review/OMISSIONS.md)、[审计代码](../../../review/OMISSIONS.md)、[冻结计划](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/plan.json)、[模型CPU关](../../../投机采样/results/runtime/p0-shared-prefix-torch-model-v1-20261012-194914/preflight/CPU-GATE.json)。
