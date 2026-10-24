---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T20:58:56.516598+08:00'
plan: PLAN.md
current: M40-DUAL-5090-HANDOFF
next: 双A100在原共享预算内自动接续RoboTwin，5分钟后台巡检；5090下一步重新核对guarded并重测Torch/guarded/自有三方eager阶段与整层，旧速度不继承。
---

# A100双卡与5090分工已生效

2026-10-23T20:58:56.516598+08:00核验：远端A100 GPU0新namespace已接续RoboTwin，GPU1原队列继续。共享队列完成148/600组、444/1800局；同148组T/C/P成功119/118/122，仅扩大DEV，四权限false。含活动租约累计32.867/160 GPU小时，global halt为空。

后续完整阶段表恢复原Torch、已有guarded、自有persistent三方；相同输入/缓存/计时边界，默认全部eager。guarded在5090未测，不能挪旧A100值。GPU0首个新pair审计待原guard自动完成；无自动重试。

[当前交接](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-continuous-20261023/RESULT.md)。



# 5090精度对齐闭合（2026-10-23T20:33:39.096761+08:00）

5090精度修复已完成：当前ffn-bias-last在三份真实B14输入（KV288/1152/2160）上26中间量＋输出逐位一致；两独立进程默认/非默认流共156完整block重复核验通过，有效KV、输入、RNG和FFN队列终态一致。修复六投影10段、FFN2七段、末段bias次序，以及原B14预计算后去重缓存。仅独立SM120研究采用；修复后性能未测，旧速度不能继承，四科学权限及production_admitted=false。

保守累计24832.512/36000秒，剩余3.102小时；20:30:12 owned清理，源码/DSO/结果已回本机。一次非法访存失败及三次数值未过保留；没有SHA、计数器、Graph或新测速。

[精度回传](../../experiments/rtx5090-block-port-20261023/precision/REVIEW_RETURN.md)、[新版27阶段表](../../experiments/rtx5090-block-port-20261023/precision/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/precision/SELECTED.json)。

下一步：在当前已对齐版本上重测原Torch双方eager三KV整层与阶段性能；旧profile二进制不对应新版，需重编译并核对输出。原累计预算不增加，不扩DAG/训练/机器人。

# 当前研究进度（2026-10-23T19:38:39.747829+08:00）

5090完整单层已实际迁移，终态CLOSED_DIAGNOSTIC_NOT_QUALIFIED。B14真实短/中/长KV288/1152/2160，两个独立进程，正常二进制双方eager原Torch比值1.2201/1.7155/2.1406×，双方Graph1.1062/1.6024/2.0361×。输出相对L2为0.9387%/0.7519%/0.6759%，最大绝对差2.5/2/2，首次分歧在Self QKV；不能称精度对齐加速。LN1/调制1、有效KV、输入/RNG门通过，仅三fixture，不继承A100证书。初轮compact-input缓存不等价；最终改为原B14输出去重，Torch缓存baseline恢复逐位一致，自有仍失败。原r1/4次构建失败保留。

170CTA×8warp、97.75KiB shared、255寄存器、静态local784B；阶段FFN联合约0.736–0.740×，Self Attention诊断约3.12/5.49/5.64×。阶段含同步，不能代替纯计算/硬件counter。所有源码/二进制/结果已回本机；19:35:16自有进程退出、显存0。保守生命周期31.929分钟，旧累计更新为23153.579/36000秒；四科学权限false，原A100采用入口不变。

[本轮回传](../../experiments/rtx5090-block-port-20261023/REVIEW_RETURN.md)、[全阶段表](../../experiments/rtx5090-block-port-20261023/STAGE-TABLE.md)、[资源](../../experiments/rtx5090-block-port-20261023/RESOURCE-AND-STOP-REPORT.md)。

下一步：本轮5090GPU作业已闭合；建议下轮先对齐当前Torch投影累加与舍入，再评估FFN，不自动扩展DAG或恢复新GPU作业。

## 此前带时戳记录（保留；安装/未测状态已被上方迁移结果更新）

# 当前研究进度（2026-10-23T19:02:24.222118+08:00）

用户新授权M39-RTX5090-PORT：单层参数/六份真实输入/当前两槽源码已迁移，SM120完整kernel正在编译；最多2小时至21:06:44，原累计不重置。无新DAG、训练或机器人改动。[本轮计划](../../experiments/rtx5090-block-port-20261023/PLAN.md)。

5090隔离环境已安装：Torch2.9.0+cu130、CUDA13.0.88、NCU2025.3.1；Torch/nvcc支持sm120，依赖与构建文件核对通过，GPU kernel0。累计安装22.50分钟，约6.05GiB，owned安装进程退出、日志/锁定清单已回本机。利用率100%原因UNKNOWN，但不阻塞适配；计数器需宿主管理员，未发生重启。两次下载调整FAILED保留，旧研究预算/BAAI队列/四false不变。 [安装结果](../../maintenance/rtx5090-environment-20261023/SETUP-RESULT.md)。

安装前只读记录：原版32GB/170SM，25 CPU额度与90GiB内存，数据盘50GiB可用。现有CUDA12.4/Torch2.5.1未配置SM120；counter受限（无SYS_ADMIN/PERFMON），util100%与0显存/空compute列表不一致，原因UNKNOWN，未启动kernel或NCU。用户条件选为后续性能基准并要求源码/结果回本机，已Maintain；尚未迁移/安装/测量。两槽候选97.75KiB仅容量可容纳，实际适配未验；旧预算、BAAI队列及四false不改。 [环境证据](../../maintenance/rtx5090-environment-20261023/RESULT.md)。

FFN1两级补充（2026-10-23T18:13:52.173911+08:00）：按用户要求在L13研究worktree仅将尾块实际输入流水3→2。完整CUDA编译＋CPU sizeof通过，Shared112384→100096 B（109.75→97.75 KiB，下降10.93%）；首轮Config特化编译失败保留，修订保留原cp.async迭代器。新版GPU精度/性能NOT_MEASURED，最近GPU通过仍为冻结L13、原使用入口A不变。CPU两次合计337.675秒、GPU0秒、owned编译器退出。原T0–T4账本/四false不改。[结果](../../experiments/single-layer-async-gemm-20261020/ffn1-two-stage-20261023/RESULT.md)。

当前任务映射见[PLAN](PLAN.md)。本轮计时的A/L1/L13/D/G全部使用外层CUDA Graph，自有算子也由脚本捕获并重放；双方eager对照未测。详见[计时条件核对](../../experiments/local-warp-pipeline-20261023/MEASUREMENT-MODES.md)，不改原始测量或开启新GPU作业。

M39-LOCAL-WARP已完成T0至T4有限判定。FFN2保留四warp MMA的7KiB局部准备B/C各225次有限门通过，诊断看到代表warp准备/MMA重叠，但正常C/A变化仅+0.082%/−0.054%/+0.104%，接近A/A波动，主线INCONCLUSIVE且不采用。B串行额外barrier也落到无准备第一段，控制不纯已记录；不能把C/B差全部归于重叠。TMPDIR首失败与误比较未初始化slot尾部的诊断失败原样保留。

副线L1有限门通过且无可信退化后扩L13；最终隔离研究保留LN1/LN3＋各自调制融合，原Welford树、FP32顺序和BF16边界不改。固定源码commit `1ef5b83af38d523b835f2511018334ba99a4b694`，source d714ebfe…、DSO164e42cb…。原两机实际使用optimized仍A（fb650be4…/a41d778d…），未自动部署生产或修改投机侧。baseline/原tile及保护文件未改。

最终同轮A/L1/L13/D/G，短中长KV288/1152/2160：L13完整单层warm Graph 0.866534/1.036672/1.049011ms；相对A延迟降0.687%/0.572%/0.428%，相对原Torch静态dense D为1.3029/1.8530/2.9994×。相对现有guarded G短快1.610%，中长仍慢3.602%/3.391%，不称全面超越强baseline。两独立进程同轮交错；未称统计显著、30层、chunk或E2E。D/G与自有同等Cross最终K/RMS/V预计算。G是Torch其余算子＋已有G1/G2共享历史attention，不是Torch官方路径或新整层persistent。

L1/L13各9真实签名81调用＋36结构144调用，输出/23共有中间量/有效KV/输入/RNG/合法B2状态有限逐位通过；最终Graph重复通过。不证明全部在线输入。数学tile2736/2808/2736，grid23→21，108CTA×8warp、shared112384B、255寄存器、静态local1040B（A1016B），1CTA/SM。动态spill未测。新版阶段表完整结构已核对；本轮阶段耗时均NOT_MEASURED，不能沿用旧插桩数值。

16:26:34本轮所有owned进程按出生身份核验退出，GPU0显存11MiB，GPU1机器人继续。保守占用窗口2809.990秒（46.833分钟），GPU workers290.455秒、CPU编译758.377秒分别列示；CUDA active UNKNOWN。旧累计18427.875秒保留，新保守累计21237.866/36000秒，剩4.101小时；本轮提前收口，不自动消费余额。远端证据约209.3MiB，低于3GiB。原截止不重置，四项科学权限及production_admitted=false。

M40只读检查点16:33：125个完整三臂组，共375个已配齐自然终态；严格同125组T102/C98/P101成功。未配齐attempt不混入分母；本轮未重审全部配对原tensor，不据此签新质量证书。GPU1既定队列继续，global_halt为空。微波炉watch16:32为20/100组、T4/C1/P2，无身份错误；按用户要求等100组齐后再查原因。GPU0本轮算子已释放，不自动重启原过期机器人窗口。原160GPU小时/10天与11/01 22:03截止、失败保留。

M17四套LIBERO6000局历史结果齐，但训练/DEV暴露、严格配对和长尾缺口保留，四项科学权限false；不恢复过期队列。质量与系统主张分开。

日期节点：10/23冻结身份与claim，10/28主表，11/01 18:00中文证据包；不足则收缩claim。用户当前无H100，H800仅候选、未接入/新授权。正式RSS2026论文已核实LingBot-VA部署RTX5880Ada、5v/10a；不可拿其绝对时延直接比本地20/50路径。

下一步：5090工具链已就绪；下一步适配SM120源码并验证实际算子，正式计时另核对争用与稳定性。利用率读数不作为环境阻塞；旧研究窗口未恢复。 原T0–T4已收口，不由环境安装扩展DAG或FA2重写。

[回传报告](../../experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[完整阶段表](../../experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[最终同轮对照](../../experiments/local-warp-pipeline-20261023/FINAL-COMPARISON.md)、[源码身份](../../experiments/local-warp-pipeline-20261023/SOURCE-MANIFEST.json)、[资源](../../experiments/local-warp-pipeline-20261023/RESOURCE-AND-STOP-REPORT.md)、[只读机器人检查点](../../experiments/local-warp-pipeline-20261023/READONLY-ROBOT-CHECKPOINT.json)。

历史完整状态：[本轮启动前](../../experiments/local-warp-pipeline-20261023/PRIOR-CYCLE-PROGRESS.md)、[闭合前](../../experiments/local-warp-pipeline-20261023/PRIOR-CYCLE-PROGRESS-before-close.md)；原始执行/失败继续保留。
