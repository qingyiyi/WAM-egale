## 2026-10-23T02:23:30.193965+08:00 M40临时双卡巡检

两卡实际运行，合计41/1800，T12/14、C12/13、P12/14；13完整pair审计通过，GPU0已通过2组。无global halt，5分钟watch继续。GPU0约10点自然停派规则不变。[检查点](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)。

## 2026-10-23T02:12:38.420042+08:00 M39六投影/QKV收口；M40临时双卡实际接续

当前单层原Torch1.2967/1.8469/2.9910×，相对上版延迟降2.211%/1.995%/1.841%。三候选各225有限门通过，大M128退化、K128编译失败保留。联合QKV减少4次grid、增量很小；新全阶段重测。累计18427.875/36000秒，02:03算子owned清理，GPU0已交接无新算子。

原GPU1完整27局后PAUSED，原子pair队列去重，两卡实际运行；合计37/1800、12完整pair审计通过，GPU0首组3/3成功。GPU0按预测约10点停派、完整episode自然终态；GPU1继续，总160小时/10天不增。完整reset bank直接引用，首0卡物理/像素/配对通过。四false保留。[结果](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESULT.md)、[全表](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[双卡检查点](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/CURRENT-CHECKPOINT.json)。

## 2026-10-23T01:21:17.786298+08:00 M39独立8warp Self QKV采用与新版完整阶段表

两版各9真实/36结构/225次有限门通过；混合函数整层慢1.72%–2.02%，隔离函数整层快1.769%/1.819%/1.438%。原Torch完整单层1.2661/1.8074/2.9322×。当前全阶段已重新计时，Self QKV仍慢于Torch；栈帧208→混合256→独立8B仅静态证据，动态local/利用率未知。首次漏函数CPU编译失败保留。两机采用，所有owned退出，累计16672.587/36000秒。本轮截止不变。GPU1自动队列19/1800，四false不改。[结果](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/RESULT.md)、[阶段表](../../experiments/single-layer-async-gemm-20261020/self-qkv-warp-control-20261023/STAGE-TABLE.md)。

## 2026-10-23T00:50:11.028665+08:00 M40第16局终态与自动接续

open_microwave-valid000-P完整到horizon1500失败，actual_commands1520与environment_take_action_cnt1500分列保留，不改原计数语义。当前T5/5、C5/5、P5/6；本组Target和连续TF尚未配齐，不能作相对效果结论。队列正在Target同case，无自动重跑，预算/质量权限不变。[检查点](../../experiments/robotwin-pilot-20261020/eval100-20261022/CHECKPOINT-20261023-0050.json)。

## 2026-10-23T00:47:13.615406+08:00 M39内联＋尾K64与当前阶段表完成

两项各225次有限调用通过，两机采用。原Torch完整单层1.2426/1.7780/2.8859×；当前阶段诊断两进程、资源一致、扰动≤0.462%，FFN合计约1.01×、小投影约0.68–0.74×。首插桩和独立FFN重放编译上下文变化保留，禁止择优改为生产独立FFN通过。GPU workers238.919秒、CPU编译904.875秒，保守2712.273秒，累计15315.883/36000秒；全部owned退出，GPU0空闲。远端NCU限制已Maintain，本机0启动。

M40 GPU1截至00:45完成15/1800、三臂各5/5；5配对审计通过，open_microwave继续，非总体质量结论。固定队列/预算/四false不改。[阶段表](../../experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/STAGE-TABLE.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESOURCE-AND-STOP-REPORT.json)。


## 2026-10-23T00:01:25.501131+08:00 M39外层FFN ABI单因素启动

用户批准先处理外层流水设备函数的寄存器保存/恢复边界，再处理FFN1尾部。仅将action_ffn_pipeline改为forceinline，当前DSO仍为旧B，GPU0有界guard CPU编译中。新窗口至10/23 00:58，累计起点12603.609944/36000秒；9真实/36结构门通过后才三KV整层计时。GPU1已完成600有效初态并自动进入模型/reset bank加载，尚无新质量结果。

[本轮JOB](../../experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/JOB.json)。
## 2026-10-22T22:19:55.013997+08:00 M39当前FFN profile完成；M40新队列准备运行

三KV两进程＋本机NCU：FFN1末组207–209μs，末FFN2约92/18/1μs计算/写回/等待；local29.8MB。生产B不改，有限门通过；累计12603.610/36000秒，owned清理，启动通道超时已单列，未重跑。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESULT.md)。

用户批准M40新160GPU小时/10天，随后纠正固定GPU0算子/GPU1机器人。GPU1官方初态补齐已运行，后续bank与完整队列自动接续，尚无新质量局。[计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/PLAN.md)。旧预算/截止/false不改。

## 2026-10-22T21:59:31.193830+08:00 M40新资源授权

用户明确同意固定1800局在现有远端两卡接续，总上限160GPU小时/最长10天；旧24小时和截止保留。先补齐manifest/bank和队列，再GPU1启动；FFN profile期间GPU0保留。该授权已Maintain，不改变质量权限。

## 2026-10-22T21:26:53.880986+08:00 M39-FFN-EPILOGUE完成

两项顺序实现：A四元素写回/常量外提，同轮完整层延迟降1.611%/1.349%/1.334%；B只内联生产函数，同轮再降1.020%/1.156%/0.811%。两次各9真实/36结构有限门通过。当前原Torch完整层1.2185/1.7605/2.8473×；独立FFN0.9182/0.9167/0.9161×，仍慢约9%，重放副本扰动≤0.24%。local1232→920B为静态属性，不冒充动态HBM。

两机研究采用B。GPU worker149.253秒、CPU编译442.124秒、保守1042.947秒；累计12338.592/36000秒，旧失败/截止和四科学权限保留。16 owned退出，本轮无失败/timeout。本机未用GPU；未开机器人。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-abi-20261022/RESULT.md)。

## 2026-10-22T19:53:45.599095+08:00 M39-FFN-TAIL完成

FFN1末4块按96有效行拆为12个M32任务。9真实/36结构有限精度与状态通过；原Torch三KV完整层1.1910/1.7241/2.7993×，同轮R2增量1.932%/1.918%/1.856%。debug支持最后组更早就绪，但FFN2尾段与static local增加，保留代价。两机研究采用；源码/DSO身份、输入一致性、资源清理收口；四科学权限false。

累计11295.646/36000秒，截止20:29:24保留，GPU worker59.630秒，保守计411.282秒，无本轮失败/timeout。未启动机器人。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/tail-row32/RESULT.md)。

## 2026-10-22T19:45:55.100647+08:00 M39-FFN-TAIL准备

用户已确认先尾部分摊，当前仅一个M32方案。108个大tile与12个尾部小tile覆盖原矩阵一次；四组计数28/28/28/36，FFN2不改。源文件直接改optimized，旧R2源码保存在现有M39证据下；新编译/验证窗口至20:29:24，累计起点10884.364/36000秒。未启动机器人或正式性能。

# 执行检查点

## 2026-10-22T18:59:32.671003+08:00 M39-FFN-DIAG完成

三KV两进程FFN子图相对原Torch0.8080/0.8095/0.8102×，有限逐位/状态通过。72条时间线一致：最后组3的生产尾部，随后FFN2计算/写回占末尾约68%/25%，提交等待约1us。嵌入与独立差不到1%。本机NCU同计算诊断路径local读27.54MB/写28.15MB；生产SASS支持主要为设备函数保存恢复，未量化各原因的可消除时间。

生成诊断副本，没有修改生产代码/DSO。三类失败保留：PID namespace、Graph捕获未replay比较、远端NCU权限；同DSO修脚本后完成，本机计数与远端性能分列。两端owned退出，GPU worker98.889秒，CPU编译147.173秒，保守含准备/清理1039.686秒，累计10884.364/36000秒，deadline19:24:12与旧截止保留。后续优化未启动。 [结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)。


## 2026-10-22T18:16:14.446638+08:00 M39-FFN-PIPE完成与清理

R1与R2分别通过9签名/36结构有限逐位与状态核验。R2保留两K分段并行、仅最终提交等待；两进程原Torch同缓存整层比值1.1646/1.6808/2.7386×，比同轮此前阶段版快约0.33%–0.36%，小幅信号，实际应用显著性未知。debug四组计数完整、提前消费实证，最后生产/消费组均3。

R2首次build-lib目录错误致0调用FAILED。保留原失败；R2产物正确安放，R1从原对象恢复且SHA256与首次测量一致，之后同截止新retry完成，不改数学代码。18:10:40自有14身份均退出；GPU生命周期136.105秒，保守896.121秒，累计9844.678/36000秒。

两机研究入口采用R2，本机未另测；ACT保留、warm数学任务不变、grid29→27。四科学权限false。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESOURCE-AND-STOP-REPORT.json)。


## 2026-10-22T18:04:28.022111+08:00 M39-FFN-PIPE有界执行

用户批准FFN1到FFN2四组依赖流水。R1已完成9签名/36结构有限逐位与状态核验、两进程三KV计时；debug确认组0/2提前消费，但完整层略慢。R1人为将第二段MMA也等到第一段提交，已在R2改为仅提交等待，并让96任务全被领取后的空闲CTA进入出口屏障。R2正在CPU编译，未采用新DSO；R1保留。

每CTA先完成全部静态FFN1生产任务，最多48个第二段CTA持有shared部分和等待；固定108常驻CTA下其余CTA仍可推进第一段。两版各200种抽象时序模拟无死锁；实际内存/数值由GPU门验证，不以模拟替代。

[JOB](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/JOB.json)：本轮45分钟，截止2026-10-22T18:40:44.193684+08:00，累计起点8948.557秒、总36000秒；R1 GPU生命周期65.188秒计入本轮。只用原Torch外部性能baseline，previous/R1仅作内部回归控制。baseline/投机/原tile/保护文件不改。


## 2026-10-22T17:44:23.726236+08:00 M39-FFN-FUSE完成并清理

直接在optimized完成FFN1+GELU、FFN2+残差3写回融合；保留BF16边界。9签名81调用与36结构144调用有限逐位/有效KV/RNG通过。两独立计时进程各5646完整层调用，相对原Torch1.1646/1.6893/2.7445×；融合本身退化1.09%/1.14%/0.83%，没有新阶段profile或warp重叠结论。

17:37:30 owned进程全部退出；GPU生命周期65.147秒，保守全窗口598.644秒，累计8948.557/36000秒。首CPU编译失败及原截止保留，r2未重置预算。两机研究入口采用已验证DSO，原版本可恢复；本机未另测。四科学权限false。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESOURCE-AND-STOP-REPORT.json)。修复当前Cycle派生导航的frontmatter和稳定ID表，不改保护文件或历史证据。

## 此前检查点（原文保留）

# 最近执行检查点（2026-10-22T16:37:20.064075+08:00）

1. 用户明确批准预计算后公平对照与完整逐阶段表，Maintain已保存；M39原累计预算不增。
2. 直接修改optimized，加入context_kv，warm跳过两投影与K RMS；baseline用外部适配器同等缓存。
3. 两项CPU构建准备失败已修复并保留；正常版编译后9签名/27结构/三KV缓存baseline逐位与状态通过。
4. 两独立进程七臂计时完成，warm自有0.957/1.124/1.140ms；强guarded仍快7.94%–11.70%。
5. 首profile因插桩遗漏入口/出口计数退出；原FAILED保留。补齐同两处计时helper，修复诊断版在同一截止内完成；正常GPU计时未重跑。
6. 两机研究入口采用；完整阶段、扰动和grid等待分别归档。最终owned清理，累计8349.913/36000秒。

[结果](../../experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)，原预算/失败/四科学权限保留。

7. 2026-10-22T16:53:49.249001+08:00 用户指出聊天表遗漏tile数量及工作量；已按当前源码补齐原STAGE-TABLE/STAGE-MAP，新增TILE-WORKLOAD.csv。区分FFN2的48输出/96分段、attention两4warp组与内部循环及实际尾块；仅CPU文档核查，无GPU消耗，原计时/失败/版本不改。

8. 按最新反馈新增STAGE-SPEEDUP.md：同轮完整层相对原Torch及强guarded比值；逐阶段可比参照缺失明确标注。仅CPU汇总，无GPU启动，Maintain更新展示要求。

9. 用户将后续对照改为仅原Torch。CPU核对GELU元素/tile数、既有原Torch grid，以及FFN epilogue融合的BF16边界和K归约依赖；已记录只读分析，无GPU消耗、无计算源码修改。

10. 新授权FFN写回融合已实现；JOB登记45分钟、18:12:32截止、原累计8349.913秒。首CPU编译因bp变量遮蔽失败（0 GPU作业），改为显式buffer指针后r2同截止继续。旧运行和源身份保留；先9签名/36结构，再原Torch三KV两进程整层对照。

## 2026-10-22T18:39:12.494633+08:00 M39-FFN-DIAG开始

用户批准先定位FFN。新诊断窗口45分钟，截止2026-10-22T19:24:12.494633+08:00；旧融合deadline不变、不恢复旧作业，累计起点9844.678秒。远端GPU0空闲，计划两CPU线程，新增512MiB。活动R2源码/DSO保持，诊断副本只作测量。清单见[PLAN](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/PLAN.md)。
