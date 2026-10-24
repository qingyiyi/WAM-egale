## 2026-10-23T19:38:39.747829+08:00 M39-RTX5090-PORT闭合

5090完整单层已实际迁移，终态CLOSED_DIAGNOSTIC_NOT_QUALIFIED。B14真实短/中/长KV288/1152/2160，两个独立进程，正常二进制双方eager原Torch比值1.2201/1.7155/2.1406×，双方Graph1.1062/1.6024/2.0361×。输出相对L2为0.9387%/0.7519%/0.6759%，最大绝对差2.5/2/2，首次分歧在Self QKV；不能称精度对齐加速。LN1/调制1、有效KV、输入/RNG门通过，仅三fixture，不继承A100证书。初轮compact-input缓存不等价；最终改为原B14输出去重，Torch缓存baseline恢复逐位一致，自有仍失败。原r1/4次构建失败保留。

原Torch同卡、同输入、同缓存，eager和Graph双方各自相同；3KV/18228完整block调用，GPU worker生命周期62.395秒、保守窗口1915.713秒，累计23153.579/36000秒。19:35:16自有PID出生身份全部退出；源/DSO/结果已回本机、源文件byte核对相同。阶段表完整27项保存r1与final；不修改原A100/投机源码或四false。

[回传](../../experiments/rtx5090-block-port-20261023/REVIEW_RETURN.md)、[全表](../../experiments/rtx5090-block-port-20261023/STAGE-TABLE.md)、[清理](../../experiments/rtx5090-block-port-20261023/OWNED-CLEANUP.json)。

## 2026-10-23T19:02:24.222118+08:00 M39-RTX5090-SETUP闭合

5090隔离环境已安装：Torch2.9.0+cu130、CUDA13.0.88、NCU2025.3.1；Torch/nvcc支持sm120，依赖与构建文件核对通过，GPU kernel0。累计安装22.50分钟，约6.05GiB，owned安装进程退出、日志/锁定清单已回本机。利用率100%原因UNKNOWN，但不阻塞适配；计数器需宿主管理员，未发生重启。两次下载调整FAILED保留，旧研究预算/BAAI队列/四false不变。 [结果](../../maintenance/rtx5090-environment-20261023/SETUP-RESULT.md)；最终清理19:00:24，SSH正常退出。

## 2026-10-23T18:38:29.559857+08:00 M39-RTX5090-SETUP开始

用户批准5090环境适配与尝试实例重启。已启动项目内独立Torch2.9.0/cu130和CUDA13.0安装，4CPU/60分钟/20GiB/GPU kernel0；原系统不改。普通reboot因容器无systemd/权限未执行，待安装闭合后由控制台重启；计数器宿主命令已整理。 远端installer PID2932/出生2618991509，02-torch PID2938/2618991764，开始18:37:06，绝对安装截止19:37:06；仅owned安装进程。

## 2026-10-23T18:23:30.445367+08:00 M39-RTX5090-ENV只读闭合

5090环境只读核查完成：原版32GB/170SM，25 CPU额度与90GiB内存，数据盘50GiB可用。现有CUDA12.4/Torch2.5.1未配置SM120；counter受限（无SYS_ADMIN/PERFMON），util100%与0显存/空compute列表不一致，原因UNKNOWN，未启动kernel或NCU。用户条件选为后续性能基准并要求源码/结果回本机，已Maintain；尚未迁移/安装/测量。两槽候选97.75KiB仅容量可容纳，实际适配未验；旧预算、BAAI队列及四false不改。 [结果](../../maintenance/rtx5090-environment-20261023/RESULT.md)。SSH正常退出，无owned计算任务；凭据未保存。

## 2026-10-23T18:13:52.173911+08:00 M39-FFN1-TWO-STAGE CPU闭合

FFN1两级补充（2026-10-23T18:13:52.173911+08:00）：按用户要求在L13研究worktree仅将尾块实际输入流水3→2。完整CUDA编译＋CPU sizeof通过，Shared112384→100096 B（109.75→97.75 KiB，下降10.93%）；首轮Config特化编译失败保留，修订保留原cp.async迭代器。新版GPU精度/性能NOT_MEASURED，最近GPU通过仍为冻结L13、原使用入口A不变。CPU两次合计337.675秒、GPU0秒、owned编译器退出。原T0–T4账本/四false不改。[结果](../../experiments/single-layer-async-gemm-20261020/ffn1-two-stage-20261023/RESULT.md)。

## 2026-10-23T16:36:22.689192+08:00 M39局部warp有界实验闭合

T0–T4完成：FFN2重叠C正常净收益INCONCLUSIVE，不采用；LN1/LN3融合L13有限225调用逐位/状态通过，完整层相对A延迟降0.687%/0.572%/0.428%，相对D1.3029/1.8530/2.9994×，中长仍慢强G3.602%/3.391%。独立源码commit1ef5b83，生产未部署。阶段本轮未插桩，完整结构/所有未测标记和失败快照均已直接聊天打印。

保守占用窗口46.833分钟，workers290.455秒、CPU编译758.377秒；累计21237.866/36000秒。16:26所有owned退出、GPU0空。两次失败保留，四科学权限false。GPU1原RoboTwin继续，只读125配齐组T102/C98/P101；微波炉20组，未启动原因分析。[回传](../../experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[全表](../../experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[清理](../../experiments/local-warp-pipeline-20261023/OWNED-CLEANUP.json)。

## 2026-10-23T14:11:40.212610+08:00 M40 GPU0中午窗口闭合核验

14:01:34完整18局/6组后停派；6组审计全通过，owned出生身份全退出、lease闭合、GPU0显存11MiB；归档与中午watch终态。raw`PARTIAL_SHARED_BUDGET_NO_NEXT_PAIR`不改，检查原guard顺序与时刻确认由本作业截止预留触发，160小时尚余133.967，无基础设施错误/截断。GPU1及主watch继续；本次无新GPU作业。

合计340/1800、113完整pair，严格同组T92/C88/P90成功；额外P成功未配齐，不混用分母。P救回6/损失4。含活跃共享26.033/160小时，余133.967；四false、全部算法失败保留。[结果](../../experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/RESULT.md)、[清理/统计](../../experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/AFTERNOON-CLOSE-20261023.json)。

## 2026-10-23T12:26:41.310598+08:00 网页版prompt按用户澄清改为独立评议

用户需要“现状＋用户想法＋网页版拿主意”，不是规定完整DAG设计任务。原prompt改为中立事实交接、保留固定源码版本与实测边界、开放方向和优先级；旧文在464bf51历史保留。Maintain保存本轮沟通偏好，不把未定方案记录为决定。无算子源码/GPU改动，RoboTwin原授权继续。

## 2026-10-23T12:13:08.831264+08:00 M40 GPU0中午新时段实际运行；DAG设计prompt交付

用户授权GPU0接续至约14:00，新namespace去重领取未开始完整pair，GPU1继续。已实际执行place_object_basket-valid016-T，teacher returned 1262、环境命令464，初态物理/观测exact。原数学/server/client源码byte相同，只改新worker编排截止；银行直接引用，CPU watcher与归档正常。原160小时含活跃/准备22.251，T0不改；当前285/1800、95完整pair，四false。阶段仍运行，不能提前写清理完成。

固定算子commit的网页版DAG设计prompt已准备，QKV交错留到统一图，静态/动态尚未定案。本轮未改算子或启动算子GPU。[启动](../../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/START-RECEIPT.json)、[计划](../../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/gpu0-midday-until14-20261023/PLAN.md)、[Prompt](../../experiments/single-layer-async-gemm-20261020/DAG-DESIGN-REVIEW-PROMPT-20261023.md)。

## 2026-10-23T11:55:56.810937+08:00 M39顶层调度讨论与顺序保存

用户将QKV交错派发留到完整DAG设计统一考虑；Maintain合并原回退/顶层规划决定，保留作用域、旧负结果与资源边界。静态图＋动态ready领取为讨论建议，未定案。本轮无计算源码改动、编译、GPU作业或新权限；RoboTwin数量仍以各自带时间戳检查点为准。

## 2026-10-23T10:09:05.210887+08:00 M40上午窗口已闭合，GPU1继续

完整256/1800、85组配对审计通过，严格同85组T69、C66、P70成功；P相对C救回6/损失2，外加1局P失败所在组未齐。GPU0完成150局/50组，于10:02:27自然结束，PID出生身份核对owned全退出，当前0%/11MiB；GPU1继续microwave-valid013-T，无global halt，CPU watch运行。原160小时累计含活跃20.075，余139.925小时；task×arm现有时长估剩余单worker约4.5–6天，不保证且不扩预算。四false、全部失败、原截止不改；本轮无新增GPU作业。[上午结果](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/MORNING-RESULT-20261023.md)。

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

## 2026-10-23T14:52:37.297603+08:00 M40等待微波炉完整后诊断

用户明确等微波炉100组三臂全部完整终态后，再核查原因；原固定队列继续，先核Target官方复现，再查TF/PF2额外失败。2026-10-23T14:52:37.297603+08:00已启动独立CPU完成提醒（每5分钟、CPU18、0GPU，PID 1001053，出生身份733681583）；首读19/100组、T4/C1/P2，无身份错误。仅自动整理终态与产物索引，因果诊断仍待完整后执行；原160小时/10天截止不变，不中途改配方。 [启动回执](../../experiments/robotwin-pilot-20261020/eval100-20261022/microwave-completion-20261023/LAUNCH-RECEIPT.json)。没有新teacher、训练、重试或算子GPU。

## 2026-10-23T20:33:39.096761+08:00 M39-RTX5090-PRECISION闭合

5090精度修复已完成：当前ffn-bias-last在三份真实B14输入（KV288/1152/2160）上26中间量＋输出逐位一致；两独立进程默认/非默认流共156完整block重复核验通过，有效KV、输入、RNG和FFN队列终态一致。修复六投影10段、FFN2七段、末段bias次序，以及原B14预计算后去重缓存。仅独立SM120研究采用；修复后性能未测，旧速度不能继承，四科学权限及production_admitted=false。

本轮保守生命周期27.982分钟，累计24832.512/36000秒；当前版9次初验＋156次公开入口确认，3真实fixture。中间失败保留、默认源码和二进制回本机、owned已清理。[精度回传](../../experiments/rtx5090-block-port-20261023/precision/REVIEW_RETURN.md)、[新版27阶段表](../../experiments/rtx5090-block-port-20261023/precision/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/precision/SELECTED.json)。

## 2026-10-23T20:58:56.516598+08:00 M40-DUAL-5090-HANDOFF

2026-10-23T20:58:56.516598+08:00核验：远端A100 GPU0新namespace已接续RoboTwin，GPU1原队列继续。共享队列完成148/600组、444/1800局；同148组T/C/P成功119/118/122，仅扩大DEV，四权限false。含活动租约累计32.867/160 GPU小时，global halt为空。

新GPU0 PID1027582，归档1027585，CPU巡检1028069；真实teacher与环境动作已推进，GPU1不改，首pair审计尚待完成。长期分工及三方eager表均Maintain。原160小时/10天/1800局、实际guard更早截止和四false不变。


## 2026-10-23T21:35:18.359044+08:00 M39-RTX5090-EAGER-STAGES登记

用户要求补逐阶段加速比，并按实际融合单元合并阶段。90分钟含准备/失败/清理，旧累计24832.512/36000秒；正常源码不改，只重建诊断版及guarded SM120比较。5090可见compute为空，预算与原科学权限不变。Maintain更新同主题constraint，不新增重复碎片。


## 2026-10-23T21:45:12.259125+08:00 M39-RTX5090-EAGER-STAGES闭合

5090当前ffn-bias-last三KV精度保持；本轮已完成原Torch/guarded/自有三方eager两独立进程计时。短/中/长Torch/自有0.8896/1.2810/1.6427倍；自有比guarded慢约27%。当前融合FFN约0.50倍、占自有阶段46%–53%；三个独立小投影0.68–0.71倍，Self QKV约1.06倍。阶段为诊断边界，非纯kernel比；无插桩整层另列，长KV guarded首进程IQR5.09%保留。

阶段表按实际融合归并为15计算阶段＋布局/退出两项，CrossKV冷构建单列；FFN重叠不拆成可加时间。正常数学源码未改，所有实验仅单层真实重放；四科学权限及production_admitted=false。新计时产物已回本机，owned退出、显存0；保守累计25426.413/36000秒，余额2.937小时。

双A100 RoboTwin仍按原160小时队列接续；本轮21:38只读确认两卡各有模型/模拟进程，没有改队列或重新统计成功率。上次完整检查点20:58为148组/444局，仅历史动态快照。

下一步与用户讨论：先隔离融合FFN的生产/等待/分段提交成本，再决定单因素优化；三个小投影第二优先。不自动扩DAG或改变数值顺序。

证据见[本轮结果](../../experiments/rtx5090-block-port-20261023/eager-stage-comparison/REVIEW_RETURN.md)。


## 2026-10-23T21:52:27.819711+08:00 M39-SM120-TMA登记

当前ffn-bias-last为正确A；用户允许TMA或其他机制优化。最多120分钟，旧累计25426.413/36000秒；TMA只改FFN2搬运，逐位通过后两进程三KV同条件eager比较。最多第二个粒度候选，主线不扩DAG。已Maintain新授权边界。

## 2026-10-23T22:12:42.902362+08:00 M39-SM120-TMA候选1终态

TMA精度有限通过，正常整层退化5.0%–5.8%，不采用；同名扩展误载计时无效并保留。第二个M64粒度候选开始编译，原截止/预算不变，当前实际保留A。[证据](../../experiments/rtx5090-block-port-20261023/blackwell-tma/reports/TMA-fixed/DECISION.json)。

## 2026-10-23T22:25:11.629514+08:00 M39-SM120-TMA闭合

5090本轮有界两候选已收口。TMA三KV逐位通过，但完整层相对A退化5.0%–5.8%，不采用；原同名扩展两次误载计时判无效、原件和预算保留。当前研究采用GRAIN-M64-FFN2：输出tile从48个128×128变96个64×128，保留四MMA warp、七段K2048及BF16提交次序。

两独立进程同轮相对A完整层延迟降18.34%/17.00%/15.91%；原Torch/自有短中长1.0864/1.5430/1.9498倍，仍比强guarded慢4.50%/5.36%/7.83%。三方全eager无Graph，冷CrossKV单列。当前融合FFN对Torch约0.75倍；三个小投影约0.68–0.70倍，余下瓶颈仍未消除。

三真实B14/Q16、block0 fixture的26中间量＋输出逐位/有效KV/输入/RNG通过；两进程前后重复、实际public默认与非默认流均通过。仅有限研究入口，四科学权限及production_admitted=false。源码/独立二进制/结果已回本机并byte核对，A保留；owned自然退出、5090显存0。完整block共22104/26000次，含7356次无效计时；保守本轮34分钟，累计27466.413/36000秒，余额2.370小时。不自动使用余额。

A100双卡继续原RoboTwin队列，22:22只读见两模型context，CPU巡检继续；本轮不重算SR、不改队列。20:58的148组/444局只是历史快照。

下一步建议先隔离剩余FFN2计算与七段提交成本，再处理三个小投影；不自动组合第三候选、扩DAG或修改数值标准。当前15计算阶段＋布局/退出，全表按实际融合边界显示，插桩分段不冒充可加的整层时间。

[回传](../../experiments/rtx5090-block-port-20261023/blackwell-tma/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/blackwell-tma/STAGE-TABLE.md)、[固定源码入口](../../experiments/rtx5090-block-port-20261023/blackwell-tma/SELECTED.json)、[原状态](../../experiments/rtx5090-block-port-20261023/blackwell-tma/HW-before-PROGRESS.md)。


## 2026-10-23T23:08:26.641496+08:00 M39-FFN-TMA-PIPE登记

用户明确同意FFN流水设计，允许容量压力下缩小tile。Maintain记录，120分钟窗口，原累计27466.413/36000秒不重置，三候选顺序门；5090显存0，当前GRAIN冻结。


## 2026-10-23T23:25:51.671858+08:00 INPUT候选收口

三KV有限逐位/状态及两进程eager完整层完成；相对GRAIN退化2.7%–3.2%，不采用。完整17行阶段表已打印并保存reports/INPUT。EPILOGUE按已授权顺序仅针对七段串行提交继续有界验证；不是采用INPUT。四false/原累计/当前public GRAIN不变。


## 2026-10-23T23:45:25.867780+08:00 M39-FFN-TMA-PIPE闭合

本轮三个FFN流水候选已完成并收口，均有限逐位/状态通过但完整层退化：INPUT约2.7%–3.2%、EPILOGUE约3.5%–4.0%、KREADY约3.5%–3.8%。当前保留GRAIN-M64-FFN2，源码与public入口已恢复；单输出槽与两输入槽放得下，未缩tile、未做尺寸网格。

最后同场原Torch/GRAIN短中长1.0892/1.5417/1.9568倍，仍慢guarded约4.03%/5.56%/7.45%。三方eager无Graph、冷CrossKV单列；候选阶段已测，当前GRAIN逐阶段为22:25历史值并明确标注，不把跨轮细微差异当因果。完整17行表逐候选已在聊天打印并归档。

三真实B14/Q16 block0的26中间量＋输出、KV/输入/RNG及两public流有限通过；KREADY14片段计数正确。一次诊断生成器编译失败保留，修复只改诊断字段；三个正常源码无数值失败。四科学权限与production_admitted=false。

自有进程自然退出、5090显存0，源码/二进制/结果回本机6份byte一致。完整block 22113/30000次；保守本轮37分钟，累计29686.413/36000秒，余额1.754小时。旧deadline不重置，不自动追加作业。

双A100 RoboTwin仍按原队列接续，23:36只读见两卡模型/模拟上下文、CPU巡检正常；本轮不重算SR、不改队列。

下一步：讨论同M64下原CUTLASS与新流水的等待/寄存器预取/提交差异，再决定是否单因素M32；容量不是本轮已证实瓶颈，不启动大DAG或无边界搜索。

[结果](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/ffn-warp-pipeline/SELECTED.json)。


## 2026-10-24T00:26:16.583833+08:00 M39-FFN-SPLITK启动

用户同意先试，90分钟且原累计十小时不增，A独立冻结。只改FFN2的K段任务组织，原GEMM、tile与数学不变；96输出tile×7段=672任务，以96个提交计数有序归并。CPU审查与320有限调度通过、负控检出死锁；5090正常编译141秒成功，三KV精度已启动。实际保留仍为GRAIN。见[计划](../../experiments/rtx5090-block-port-20261023/ffn-splitk/PLAN.md)。


## 2026-10-24T00:45:50.961788+08:00 M39-FFN-SPLITK闭合

FFN2有序split-K本轮完成并研究采用。M64×N128/K64、四MMA warp、原cp.async主循环和七段BF16提交不变，仅将96输出tile×7段变为672计算任务；170个常驻CTA领取，96计数器按序提交，新增384B全局计数。

三真实B14/Q16/block0的26中间量＋输出/有效KV/输入/RNG及public默认/非默认流有限逐位通过。两独立进程同轮原Torch/自有短中长1.1493/1.6245/2.0497×，相对冻结GRAIN完整层降5.404%/5.174%/4.474%；对Guarded短快1.54%、中慢0.22%、长慢2.56%。全eager无Graph，冷CrossKV另列。

独立FFN2＋bias＋门控残差近原Torch，较串行快约1.43×；纯Linear未对齐，完整融合FFN仍慢15%–16%。三个小投影仍约0.67–0.71×。完整17阶段本轮重测并在对话打印；插桩与正常完整层分开。TMA未启用，旧三个TMA流水负结果和GRAIN独立A均保留。

正常/profile/probe构建及七GPU进程均成功，无GPU失败/超时；7389完整block、5094独立FFN2。owned20身份全部退出，5090显存0；六份源码/二进制回本机byte一致。保守计30分钟，累计31486.413/36000秒、余额1.254小时，旧deadline不重置。四科学权限和production_admitted=false。

双A100 RoboTwin00:40只读见两卡模型/模拟上下文、CPU巡检正常；不改队列、不重算SR。下一步建议有界处理Self/Cross输出与Cross Q的有序K分段，再隔离TMA预取；本次不追加作业。

[结果](../../experiments/rtx5090-block-port-20261023/ffn-splitk/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/ffn-splitk/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/ffn-splitk/SELECTED.json)。


## 2026-10-24T01:27:17.355600+08:00 M39-PROJECTION-SPLITK登记

仅三个小投影K段任务化；冻结现有FFN2版本。60分钟、累计31486.413/36000秒，四false和原截止保留，先逐位后计时。


## 2026-10-24T01:40:36.304385+08:00 M39-PROJECTION-SPLITK闭合

本轮三个小投影有序split-K已完成并拒绝：两进程完整层相对A退化42.700%/40.873%/38.087%。候选26中间量＋输出/输入/KV/RNG及计数均有限通过，静态资源与A相同；退化集中于三个投影，尚未定量区分领取/等待/发布开销。候选及负结果保留。

当前实际保留SPLITK-M64-FFN2，已恢复五源文件及原public入口；默认/非默认流再次通过，六份源码/DSO回本机byte一致。当前正常完整层原Torch/自有1.1505/1.6270/2.0413×；相对guarded短快1.889%、中快0.171%、长慢2.987%。全部eager，无Graph，同缓存。完整17阶段已本轮重测：三个投影约0.67–0.70×，融合FFN0.864/0.858/0.865×；当前和拒绝候选各自完整表保存、在对话打印，非历史混填。

本轮11469/15000完整block，两个构建/六GPU进程成功、无timeout，性能拒绝1个。owned16身份退出、5090显存0；保守计15分钟，累计32386.413/36000秒，余1.004小时，旧deadline不重置。四科学权限和production_admitted=false。

小投影每K段仅5次K64迭代（末段3次），比FFN2每段32次短；细粒度调度难摊薄开销是待定量机制，不能说已确定atomic或fence占比。下一步建议先分量诊断，再考虑每任务多段的一个粗粒度对照；尚未启动。

双A10001:28只读检查模型/模拟上下文和CPU巡检仍在原RoboTwin队列，本轮不改队列、不重算成功率。

[结果](../../experiments/rtx5090-block-port-20261023/projection-splitk/REVIEW_RETURN.md)、[当前完整表](../../experiments/rtx5090-block-port-20261023/projection-splitk/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/projection-splitk/SELECTED.json)。


## 2026-10-24T09:59:19.785913+08:00 M39-PROJECTION-TMA-CONTINUOUS登记

只试跨十段输入ring，8MMA warp/原BF16后处理保持；50分钟且累计32386.413/36000秒不重置，正确A冻结。


## 2026-10-24T10:09:30.649000+08:00 M39-PROJECTION-TMA-CONTINUOUS闭合

连续TMA输入单候选已完成并不采用：两个独立进程完整层相对A慢1.384%/1.125%/1.065%。三投影保持96输出任务、8MMA warp；双K64输入槽跨十个数值段，原BF16合并不变。26中间量＋输出/KV/输入/RNG及每tile48A/48B/48等待/10提交有限通过。共享内存仅增256B至100352B，驻留及静态寄存器/local属性不变；实际重叠比例与各开销贡献未知。后处理warp重叠尚未实现。

当前实际保留SPLITK-M64-FFN2，五源文件/public入口已恢复；默认/非默认流通过，六份源/DSO回本机byte一致。当前原Torch/完整层1.1512/1.6240/2.0417×，guarded短快1.485%、中慢0.300%、长慢2.979%。当前17阶段本轮重测，候选另存完整表；全eager无Graph、同缓存，阶段诊断不冒充可加完整层。

两个构建、六GPU进程成功，无失败/timeout；11469完整block，性能拒绝1个。16owned身份退出、5090显存0，实例未关机。保守计15分钟，累计33286.413/36000秒，余45.2分钟，旧预算/截止保留。四科学权限与production_admitted=false。

下一步建议先分量核对TMA输入等待、局部同步与后处理，再决定输出交接；未启动新候选。双A10010:02只读见原RoboTwin模型/模拟与巡检正常，不改队列、不重算成功率。

[结果](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/REVIEW_RETURN.md)、[当前全表](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/SELECTED.json)。


## 2026-10-24T10:36:00.234691+08:00 M39-PROJECTION-TMA-COMPONENTS登记

只诊断COARSE/FINE/扰动，不改生产或采用代码；35分钟含旧累计33286.413秒，四false不变。


## 2026-10-24T10:48:11.109897+08:00 M39-PROJECTION-TMA-COMPONENTS闭合

TMA分量诊断已完成，当前SPLITK-M64-FFN2五源＋DSO六份byte未改。r0 FINE局部扰动14.62%超限保留；唯一r1修正后OFF/COARSE/FINE完整层扰动最大4.48%/4.73%/6.26%，FINE局部约5.04%，达到原筛查。四进程三fixture数学/输出/KV/输入/RNG通过，无新优化或后处理warp交接。

软件观察每tile约70微秒，其中主循环包络约57；BF16/bias/写回约7.5，加结果落shared和同步约9（观察总时长13%）。细项观察operand/转换/MMA约36.5，TMA发起约9，输入轮询4–5，输入同步合计约6。插桩仍有扰动，不报为原正常实现精确stall/纯TC/带宽比例，包络和细项不重复相加。下一步建议保留8MMA先查输入发起/交接，再决定输出分工；尚未启动。

本轮正常A原Torch比1.1573/1.6260/2.0374×；原TMA仍慢约1.14%–1.19%。完整17阶段沿10:06历史值明确标注本轮未重测，正常整层为本轮新测。四科学权限与production_admitted=false。

5946/6500完整block，两个构建/四GPU进程成功；r0计时筛查失败保留，修正1次。12owned身份退出、5090显存0；保守15分钟，累计34186.413/36000秒，余30.2分钟，旧起点/截止不重置。双A10010:43仍原RoboTwin模型/模拟和巡检，本轮不改队列或重算SR。

[结果](../../experiments/rtx5090-block-port-20261023/projection-tma-components/REVIEW_RETURN.md)、[分量表](../../experiments/rtx5090-block-port-20261023/projection-tma-components/COMPONENTS.md)、[完整阶段表](../../experiments/rtx5090-block-port-20261023/projection-tma-components/STAGE-TABLE.md)。

## 2026-10-24T12:22:08.565998+08:00 M39-PROJECTION-WARP-SPECIALIZED闭合

四MMA＋一TMA＋三epilogue有限逐位/public两流通过，原Torch完整层1.2101/1.7059/2.1253×，相对A下降4.843%/4.447%/3.984%，研究采用。原A/原TMA保留，14owned退出，六文件byte相同。完整表与资源见projection-warp-specialized；四false不变。原累计现35566.413/36000秒。

## 2026-10-24T12:28:23.195321+08:00 M40双卡基础设施恢复

已聚焦WAM Cycle。旧pair281因STATUS.tmp原子替换竞态触发HALT，原848命令中断保留。新双worker已实际推进，首两组三臂审计通过。尾部一次C补齐和每60秒CPU巡检已登记启动，600有效组才算完成。原预算不重置。证据见resume-after-storage-race-20261024。


## 2026-10-24T13:05:13.797164+08:00 M39-RESULT-EMPTY闭合

用户要求检查result_empty及TMA写回。独立观察副本一次构建/246完整block；三KV有限逐位、KV/输入/RNG通过，静态资源同当前正常，完整层扰动约1%内。empty原始单次P50/P90=32/64ns，首段无等待控制P50=32ns，九次合计约0.37μs，对应后九段warp包络约37.4μs；不报为整层硬件stall百分比。当前观察不支持优先扩epilogue warp，当前版本不改。TMA不能直接读寄存器，shared→global可研究但未实现。

源码/二进制七项byte未变，无SHA；独立构建与GPU worker正常退出、compute context清零，生命周期含回传145.891秒，保守180秒，累计35746.413/36000秒。两条CPU核对命令失败保留，最终r2核对通过。完整17阶段本轮使用明确历史值并保存快照；四false与production_admitted=false。证据：projection-warp-specialized/result-empty-diagnostic/REVIEW_RETURN.md。


## 2026-10-24T13:30:41.849615+08:00 M39-CURRENT-MMA-MAINLOOP闭合

用户已批准拆解当前四MMA主循环，Maintain保存本次范围；独立副本一次构建/1083完整block。三KV有限逐位/状态与粗细扰动门通过；稳态ready P50/P90=32/64ns，初两块有初始化等待；交错加载/转换/MMA两半合计约0.73–0.74μs/K64。W0等empty约0.65μs，TMA发起约0.17–0.18μs。不能称纯Tensor Core时间、LDSM完成延迟或整层硬件stall占比；观察局部包络增加4.8%–5.85%。当前采用版不改，17阶段表比值沿明确历史，完整表保存并在对话交付。

一次构建135.707秒、GPU进程6.724秒，含回传145.681秒、保守180秒；累计35926.413/36000秒，余73.587秒。源/二进制七项byte未变、无SHA，owned退出且compute context清零，四false/production_admitted=false。证据：projection-warp-specialized/mma-mainloop-diagnostic/REVIEW_RETURN.md。双A10013:24巡检293有效完整组、两worker继续，无新HALT；本轮未改机器人队列。


## 2026-10-24T13:49:21+08:00 M39-SELF-QKV-TMA登记

用户明确追加现有RTX 5090持续使用，取消后续作业原十小时累计上限；旧35926.41251785547秒账本原样保留，Maintain保存适用范围。双A100仍按原160GPU小时/10天预算，13:44巡检296有效完整组、两worker运行。

本job20分钟、4CPU、1GiB、12000完整block上限；冻结WARP-SPECIALIZED-4MMA独立A。Self Q/K/V仅迁移一TMA／三epilogue／四MMA，不改288任务/tile/K分段/舍入；精度先于四臂两进程计时，公开入口两流在采用时核验。5090启动时compute context为空，远端与本地A六文件byte一致，不做SHA。计划与实际终态落点：projection-warp-specialized/self-qkv-tma-migration。


## 2026-10-24T14:01:25.329383+08:00 M39-SELF-QKV-TMA闭合

用户追加现有5090持续使用已Maintain，旧十小时账本不改。Self QKV单变量迁移研究采用：27中间量/输出、KV/输入/RNG、六投影计数、公开两流有限逐位通过。四臂两独立进程同轮原Torch/新版短中长1.2475/1.7553/2.1872×，比旧A延迟下降2.942%/2.808%/2.249%，Guarded/新版1.1021/1.0850/1.0433×。全eager/no Graph，同缓存，单层非E2E。

正常/profile两构建及五GPU进程成功、无timeout；2871完整block。生命周期含等待/回传434.383秒，保守480秒，新跟踪累计36406.413秒（新授权无旧总上限）；CUDA active未知。9自有身份/进程组均退出，GPU context清零，实例未关机。六源/DSO回同步本机byte一致，无SHA。完整阶段表和采用前后快照已保存；同步口径补充为TMA本体3＋领取/完成2=每成功tile显式5次CTA同步，另有末次空领取。四false及production_admitted=false。

实际源码：.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu；固定证据：projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md、STAGE-TABLE.md、SELECTED.json。旧A及历史失败不改，未动baseline/投机/protected源。双A10013:55巡检298有效完整组，两worker继续原RoboTwin队列。


## 2026-10-24T14:08:29.933406+08:00 本轮公开快照

当前研究源码/结果、历史负结果与HW已推main，远端实查为459b86562a2f8b1ca075402bf096416e4cd83267；未上传凭据/权重/tensor/DSO/部署连接脚本。只同步WAM主题，不改本地其他Cycle。原科学门不变，publication收据见本轮PUBLISH-CHECKPOINT.json。5090 owned SSH master已正常关闭，实例未关机；GPU进程此前已清理。


## 2026-10-24T17:09:23.828654+08:00 M39-FFN-HANDOFF登记

用户明确同意四类FFN方案实际有限实验，Maintain保存本轮范围；冻结当前Self-QKV-TMA独立A。5090空闲、远端43GiB，本机空间紧张仅回传必要源/结果。B/C各30分钟/4CPU/1.5GiB/12000完整block，先精度再计时、逐候选全表，不动baseline/投机。双A10017:05为329有效三臂组、两worker继续。证据ffn-local-handoff-20261024/PLAN.md。


## 2026-10-24T17:24:18.399600+08:00 M39-FFN B/C阶段结果

B仅TMA输入分工、C跨任务单结果槽重叠，均真实三KV有限逐位/状态通过，两进程完整层均退化，当前A不改。B短中长+1.201%/+1.129%/+0.833%；C+0.900%/+0.603%/+0.722%。C新增672任务计数通过，不声称已量化真实重叠时长。完整阶段表已分别在聊天打印并保存snapshots。D初次device Gemm模板M16线程映射编译失败、0 GPU调用保留，已独立r1改warp原语重编。运行时当前kernel size2 cluster报告85活跃组，尚未完成DSM实际验证。证据ffn-local-handoff-20261024。


## 2026-10-24T17:43:43.443882+08:00 M39-FFN-HANDOFF闭合

四方案全部真实运行，三KV有限逐位/状态通过，两个进程整层及最后同轮FFN局部对照均无收益，保留A不变。B/C整层约+0.6%–1.2%，D2.03–2.26倍、E2.75–3.12倍A；小M造成逻辑权重tile请求增长，非HBM计数。双CTA DSM实际可用但本映射慢，保留负结果。13380完整block，生命周期和1207.323秒，1编译失败、0timeout，33身份及阶段组退出，context为空，实例开。完整表逐项打印/保存，最终A表本轮新测；资源与证据ffn-local-handoff-20261024。四false和production_admitted=false，baseline/投机/protected不动。RoboTwin17:39为335有效完整三臂组、双worker运行，原预算不变。
