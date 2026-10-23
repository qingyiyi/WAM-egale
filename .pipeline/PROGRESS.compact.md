---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T16:36:22.689192+08:00'
current: M39-LOCAL-WARP-complete;M40-running
next: 带回局部warp证据讨论；GPU1原RoboTwin队列继续，微波炉100组齐后查因。
---

# 当前研究进度（2026-10-23T16:36:22.689192+08:00）

M39-LOCAL-WARP已完成T0至T4有限判定。FFN2保留四warp MMA的7KiB局部准备B/C各225次有限门通过，诊断看到代表warp准备/MMA重叠，但正常C/A变化仅+0.082%/−0.054%/+0.104%，接近A/A波动，主线INCONCLUSIVE且不采用。B串行额外barrier也落到无准备第一段，控制不纯已记录；不能把C/B差全部归于重叠。TMPDIR首失败与误比较未初始化slot尾部的诊断失败原样保留。

副线L1有限门通过且无可信退化后扩L13；最终隔离研究保留LN1/LN3＋各自调制融合，原Welford树、FP32顺序和BF16边界不改。固定源码commit `1ef5b83af38d523b835f2511018334ba99a4b694`，source d714ebfe…、DSO164e42cb…。原两机实际使用optimized仍A（fb650be4…/a41d778d…），未自动部署生产或修改投机侧。baseline/原tile及保护文件未改。

最终同轮A/L1/L13/D/G，短中长KV288/1152/2160：L13完整单层warm Graph 0.866534/1.036672/1.049011ms；相对A延迟降0.687%/0.572%/0.428%，相对原Torch静态dense D为1.3029/1.8530/2.9994×。相对现有guarded G短快1.610%，中长仍慢3.602%/3.391%，不称全面超越强baseline。两独立进程同轮交错；未称统计显著、30层、chunk或E2E。D/G与自有同等Cross最终K/RMS/V预计算。G是Torch其余算子＋已有G1/G2共享历史attention，不是Torch官方路径或新整层persistent。

L1/L13各9真实签名81调用＋36结构144调用，输出/23共有中间量/有效KV/输入/RNG/合法B2状态有限逐位通过；最终Graph重复通过。不证明全部在线输入。数学tile2736/2808/2736，grid23→21，108CTA×8warp、shared112384B、255寄存器、静态local1040B（A1016B），1CTA/SM。动态spill未测。新版阶段表完整结构已核对；本轮阶段耗时均NOT_MEASURED，不能沿用旧插桩数值。

16:26:34本轮所有owned进程按出生身份核验退出，GPU0显存11MiB，GPU1机器人继续。保守占用窗口2809.990秒（46.833分钟），GPU workers290.455秒、CPU编译758.377秒分别列示；CUDA active UNKNOWN。旧累计18427.875秒保留，新保守累计21237.866/36000秒，剩4.101小时；本轮提前收口，不自动消费余额。远端证据约209.3MiB，低于3GiB。原截止不重置，四项科学权限及production_admitted=false。

M40只读检查点16:33：125个完整三臂组，共375个已配齐自然终态；严格同125组T102/C98/P101成功。未配齐attempt不混入分母；本轮未重审全部配对原tensor，不据此签新质量证书。GPU1既定队列继续，global_halt为空。微波炉watch16:32为20/100组、T4/C1/P2，无身份错误；按用户要求等100组齐后再查原因。GPU0本轮算子已释放，不自动重启原过期机器人窗口。原160GPU小时/10天与11/01 22:03截止、失败保留。

M17四套LIBERO6000局历史结果齐，但训练/DEV暴露、严格配对和长尾缺口保留，四项科学权限false；不恢复过期队列。质量与系统主张分开。

日期节点：10/23冻结身份与claim，10/28主表，11/01 18:00中文证据包；不足则收缩claim。用户当前无H100，H800仅候选、未接入/新授权。正式RSS2026论文已核实LingBot-VA部署RTX5880Ada、5v/10a；不可拿其绝对时延直接比本地20/50路径。

下一步：携回REVIEW_RETURN及全表讨论中长KV强G差距和是否止损FFN2局部准备；本轮不展开新DAG、FA2重写或跨硬件。RoboTwin沿原队列/CPU巡检运行。

[回传报告](experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[完整阶段表](experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[最终同轮对照](experiments/local-warp-pipeline-20261023/FINAL-COMPARISON.md)、[源码身份](experiments/local-warp-pipeline-20261023/SOURCE-MANIFEST.json)、[资源](experiments/local-warp-pipeline-20261023/RESOURCE-AND-STOP-REPORT.md)、[只读机器人检查点](experiments/local-warp-pipeline-20261023/READONLY-ROBOT-CHECKPOINT.json)。
