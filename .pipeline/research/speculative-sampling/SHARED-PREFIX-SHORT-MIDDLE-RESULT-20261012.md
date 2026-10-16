# 共享KV：短档与中等长度原Torch补测结果

> 2026-10-12T21:59:02+08:00 | 投机采样 | zh-CN | Asia/Shanghai

补测完成。中档与既有长档的六组对照都更快；短档中位更快，但出现一组回退，**目前不能确认短KV稳定加速，也不能宣称任意KV长度都加速**。两档全部660个计时输出与同档原始Torch逐字节一致。新监督器正常收尾，公共生产代码和受保护状态未改。

## 三档结果

| 条件 | 完整KV长度 | 每臂自然query数 | 原Torch R中位（秒） | 共享S中位（秒） | 中位耗时下降 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 短 | 288 | 27 | 3.114379 | 2.748547 | 11.75% | 混合；5/6组更快，第1组回退5.56% |
| 中 | 1152 | 28 | 3.834371 | 3.290483 | 14.18% | 正向pilot；6/6组更快 |
| 长（既有） | 2160 | 29 | 5.170557 | 3.359750 | 35.02% | 正向pilot；6/6组更快 |

这是相同条件内R/S的完整30层verifier查询序列对照。短/中/长分别有27/28/29个真实变化query，因此不同档总耗时差不能全归因于KV长度。中、长档结论只覆盖这几个固定真实条件，尚无任务成功率或E2E结论。长档来自[既有报告](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)，其父监督器退出FAIL原样保留，已完整完成的测量由独立CPU核验支持；本轮没有重跑长档。

| 条件 | R额外峰值（MiB） | S额外峰值（MiB） | 额外峰值下降 |
|---|---:|---:|---:|
| 短 | 257.373 | 257.071 | 0.12% |
| 中 | 535.827 | 257.150 | 52.01% |
| 长（既有） | 891.035 | 257.085 | 71.15% |

额外峰值为`max_memory_allocated - baseline_allocated`，不等于总驻留显存。短档约257MiB的峰值基本不变；中档下降52.01%，长档下降71.15%。共享历史不需要为14候选物化完整历史KV，但不由此声称DRAM读取减少七倍。

## 两档全部六组计时

| 条件 | 组 | 执行顺序 | R（秒） | S（秒） | S/R | 耗时变化 |
|---|---:|---|---:|---:|---:|---:|
| 短 | 1 | R→S | 3.163210 | 3.339036 | 1.055585 | +5.56% |
| 短 | 2 | S→R | 3.065548 | 2.557670 | 0.834327 | -16.57% |
| 短 | 3 | R→S | 3.008009 | 2.685808 | 0.892886 | -10.71% |
| 短 | 4 | S→R | 3.637356 | 2.932355 | 0.806178 | -19.38% |
| 短 | 5 | R→S | 3.176829 | 2.774480 | 0.873349 | -12.67% |
| 短 | 6 | S→R | 2.986653 | 2.722614 | 0.911594 | -8.84% |
| 中 | 1 | R→S | 3.800609 | 3.540828 | 0.931648 | -6.84% |
| 中 | 2 | S→R | 3.750978 | 2.949335 | 0.786284 | -21.37% |
| 中 | 3 | R→S | 4.023450 | 3.179357 | 0.790207 | -20.98% |
| 中 | 4 | S→R | 3.868133 | 2.982580 | 0.771065 | -22.89% |
| 中 | 5 | R→S | 3.715497 | 3.424904 | 0.921789 | -7.82% |
| 中 | 6 | S→R | 3.896031 | 3.401610 | 0.873096 | -12.69% |

两档均保留全部样本，无删除、替换或GPU重跑。短档R/S MAD/median为2.710%/4.485%，中档为1.891%/5.847%，均通过预声明≤7%的测量波动门。

正向pilot仍要求中位S/R≤0.95且六组均<1；短档首组S/R=1.055585，因此为`flat_or_mixed`。`PASS_STABLE_PERFORMANCE_MEASUREMENT`表示计时、数值及资源门通过，**不等于两档都达到正向加速门**。

## 原Torch输入与精度

原Torch SDPA的实际FlashAttention路径仍是金标准，同一训练模型、A_off、selector/threshold/seed、BF16、A100、batch14/CFG2/每CFG七候选，text CPU、VAE CUDA、FSDP关闭。原CUDA v2、adapter及以前失败证据均冻结，未改参考backend或放宽容差。

短档恢复真实T-init3-chunk1/frame4快照：请求前历史144，视频更新后历史272＋当前16＝KV288，50次非terminal capture产生27个真实变化候选query。中档恢复同一起点，沿原入口回放frame4、8、12、16、20、24保存的视觉terminal、动作及观察，18次cache前向建立chunk7/frame28的请求前真实历史1008；视频更新后历史1136＋当前16＝KV1152，51次非terminal capture产生28个query。动作仍经过原postprocess→preprocess。没有裁剪长缓存、随机合成历史或新增环境运行。

R/S各对该档全部query预热；六组各臂覆盖完整30层及全部真实query。直接继承冻结的原Runtime.sequence：输入staging、lease/setup/keys、写前和返回后fsync journal、release均在wall计时内；历史恢复、CPU输出拷贝/逐字节比较及存盘在计时外。OFF期间没有profiler或归因hook。

每个实际输出重新对齐同档原Torch gold；独立CPU加载24份正式输出文件，核验短324＋中336＝660个张量，共4,435,200元素，dtype/shape/finite与全部字节一致，changed_elements=0。all30 history/metadata/RNG、lease释放、额外峰值≤12GiB、free≥8GiB及清理门保留。此前全30层QKV/attention、29输出、12类失效及13行隔离证据冻结复用，没有减少正式门。

## 短档诊断与能解释的范围

短档没有达到正向pilot，按预声明方案在全部OFF结束后，用首个query为R/S各追加一次profile，额外2次前向。该诊断包含冷lease，不能代表27query摊销性能，更不能替换首组OFF回退。单query诊断wall为R 207.954ms、S 215.602ms；CUDA event为207.747ms、210.590ms，event span包含主机提交间隙，不是kernel busy。

| 诊断统计 | R | S |
|---|---:|---:|
| 关联设备事件时间并集（ms） | 103.854 | 95.409 |
| 首末设备事件之间未覆盖时间（ms） | 98.771 | 111.087 |
| CPU verifier scope（ms） | 207.348 | 210.429 |
| 30个self-attention核时长之和（ms） | 2.257 | 2.579 |
| self-attention CPU `aten::cat` 次数 | 60 | 0 |
| self-attention CPU `aten::index` 次数 | 60 | 0 |
| self-attention CPU `aten::nonzero` 次数 | 30 | 30 |
| self-attention CPU `aten::unique_dim` 次数 | 0 | 0 |

通过CUDA launch correlation及CPU模块scope关联4232个设备事件；全trace未匹配0个，scope之外主要包括缓存恢复与输出检查，未计入表中。两臂仍各有30个原cross-attention Flash核；S仅替换30个self-attention核。诊断中的共享self-attention核总时长2.579ms，比原核2.257ms高14.24%，同时省去self-attention的60次历史KV拼接及其复制。

这支持一个有限解释：短历史的复制成本较小，而共享寻址、slot索引和row校验成本仍存在，收益容易被抵消。30层索引只在lease内复用，单冷query的setup未充分摊销。设备事件之和不是关键路径，CPU inclusive时间有嵌套；profile也会扰动主机提交与同步。**第一组OFF回退的唯一原因仍未确定**，不能据此断言来自冷kernel、精度问题、某一次CPU停顿或GPU竞争。全部query均已按原计划完整预热。

CPU归因脚本第一版把全模型`aten::cat`数量误当成self-attention范围数量，断言失败；全模型R/S实际63/3次，其中self-attention为60/0次。原脚本及失败log保留，第二版按模块scope验证后通过；没有修改GPU原始结果、测量实现或计时样本。

## Prompt及模态共享范围

目前共享的是14候选读取的原CFG2历史self-attention KV池。已有视觉和动作历史处于同一历史池中，候选直接引用该历史；本轮没有新增模态路径。文本prompt投影、cross-attention K/V以及当前候选Q/K/V仍按原batch14执行，**尚未实现文本prompt计算共享**。因此这些开销仍会限制完整verifier的收益；此范围也由冻结adapter及两臂诊断中的cross-attention事件支持。

## 预算、监督器与保存

用户明确同意累计target硬上限1000→2200；新回执绑定plan SHA256 `3a1b06dd83257fca7f0e47c764249f5878946d12e26de210f55cd442ad8b4a32`。旧1000/900计划、回执与失败均不改。新计划上限1199次target、1worker/1GPU启动/无重试、900秒含加载、历史回放、IO、同步和清理；累计GPU上限仍2700秒。

实际新增897次写前意图/897次主机forward返回，0failed/0inflight/0直接attention；短433次、中464次，持久journal为CLOSE。本轮GPU 370.813737秒，累计1865/2200及2007.138085/2700秒（33.45/45分钟）；剩余335次及692.861915秒。这些余量不自动授权新类型实验。

19项CPU预检与独立准备审计通过；测量后独立CPU逐字节、六组统计、journal、2438个冻结源/数据/证据SHA及文件stat、进程/NVML/端口检查通过。退出身份修复已接入本轮新私有监督器；新父监督器正常PASS并完成清理。现场没有记录到terminal-root分支，因此仅声称新监督器正常完成与CPU分支测试通过，不声称本次实际触发了僵尸态分支。旧长档退出FAIL保持原状态。

0episode、0EGL、0完整policy chunk、0环境步、0训练。公共模型/server/config及受保护state/rules/cycle未改。实现与执行审查由同一主代理完成，独立CPU脚本复查不称为独立代理审查。HW继续采用chat记录，不创建新Milestone、Patch或Cycle。

## 剩余问题与下一步

短KV仍不能保证稳定加速；优先针对短档的slot/row校验、索引生命周期和KV访存成本设计单独的CPU方案，保持原Torch逐位门，再制定有界验证。当前不自动改kernel、加测、实现prompt共享或启动任务闭环。任务成功率/接受率/E2E仍待有独立方案与预算的真实任务验证。

当前状态：`SHORT_MIXED_MIDDLE_LONG_POSITIVE_VERIFIER_TASK_PENDING`。数值对齐与本轮补测完成；短档稳定收益、文本prompt计算共享、真实任务成功率/接受率及E2E仍未完成。

## 复查入口

- [本轮最终GPU结果](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/FINAL_RESULT.json)、[短档OFF统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/short-OFF-RESULT.json)、[中档OFF统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/middle-OFF-RESULT.json)、[独立CPU审计](../../../review/OMISSIONS.md)。
- [短档profile归因](../../../review/OMISSIONS.md)、[原始短档trace](../../../review/OMISSIONS.md)、[真实中档历史回放](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/results/middle.prefix.json)。
- [完整方案](../../../投机采样/experiments/shared_prefix_torch_lengths_v1/PLAN.md)、[冻结计划](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/plan.json)、[预算回执](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/BUDGET-AUTHORIZATION.json)、[19项CPU门](../../../投机采样/results/runtime/p0-shared-prefix-torch-lengths-v1-20261012-213536/preflight/CPU-GATE.json)、[准备审计](../../../review/OMISSIONS.md)。
- [原Torch数值对齐报告](SHARED-PREFIX-TORCH-ALIGNMENT-RESULT-20261012.md)、[既有长档报告](SHARED-PREFIX-TORCH-PERFORMANCE-RESULT-20261012.md)。
