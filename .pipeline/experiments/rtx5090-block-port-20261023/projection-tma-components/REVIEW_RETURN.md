# 连续TMA分量诊断：完成，当前代码不变

终态 **CLOSED_DIAGNOSTIC_COMPLETE_RETAINED_UNCHANGED**。仅诊断r0/r1，当前继续SPLITK-M64-FFN2，未实施后处理warp交接。

## OBSERVED

两个版本各两个独立进程，在三份真实B14/Q16/block0输入上完整执行。原Torch、Guarded、当前A、原TMA正常、诊断OFF/COARSE/FINE全部eager无Graph、同硬件/缓存；所有中间量/输出/有效KV/输入/RNG及原FFN/TMA提交门通过。

r0 FINE局部扰动最高14.62%，超过登记10%，保留为不能用于原实现定量归因。唯一一次r1修正把相邻end/begin两次读钟改为一次，数学和同步不变。r1完整层OFF/COARSE/FINE最大扰动4.48%/4.73%/6.26%，FINE/COARSE局部最大5.04%（各进程最大约5.19%），满足预登记筛查。静态资源与原TMA同为170CTA/1CTA每SM/100352Bshared/255reg/784Blocal/256线程；不代表动态行为相同。

r1粗计时每输出tile约69.4–70.4微秒；主循环包络57.1–57.8微秒。BF16合并/bias/写回约7.35–7.63微秒，加结果落shared及前后同步合计约9.1–9.4微秒，占这个观察者总时长约13%。初始化/清理/循环及记账间隙另列，没有把间隙藏进MMA。

r1细计时观察：input_poll约4.39–5.20微秒，ready同步约3.49–3.52，operand/shared→register/转换＋MMA约36.35–36.65，reader同步约2.60–2.65，TMA发起约8.96–9.10。具体九组和原始trace在COMPONENTS.md、SUMMARY.json及四个RESULT.json。

这些是thread0观察者路径上的软件区间，不是原实现精确stall或纯TensorCore/GPU流量。FINE仍使局部时长增加约5%，诊断OFF本身也慢于原正常版；数据适合定位方向，不能把百分比直接写为未插桩代码真实构成。尤其TMA_enqueue包含观察者交界的记账开销，不等于整个异步传输耗时。主循环包络与细项有包含关系，禁止重复相加。多个tile工时不能相加冒充阶段或完整层延迟。

同轮未插桩正常A相对原Torch为1.1573/1.6260/2.0374×；原TMA正常相比A慢1.179%/1.186%/1.142%，复现前轮无收益。r0正常主表16轮×5；r1诊断修正10轮×4，身份/预算/样本不混用。

## HYPOTHESIS / UNKNOWN / RECOMMENDATION

观察支持主循环的operand/MMA及发起/交接值得优先处理，后处理有约9微秒的工程窗口，但不是唯一可优化部分。该窗口完全隐藏只是理想分析，未完成真实交接实验，不能保证整层收益。原8MMA warp改为更少warp可能损失计算吞吐，因此不建议仅凭约13%的观察占比立即拿走MMA warp专职后处理。

下一步建议先保持8MMA，核查输入槽释放点、TMA发起与CTA交接能否更早或更少；量化后再设计最小输出交接对照。TMA等待是否被完全隐藏、哪类硬件stall主导、HBM/TensorCore利用率均UNKNOWN；没有可用硬件计数器，不补造。未启动新的优化作业。

## 交付、资源与当前完整表

诊断源码/两版DSO/四进程原始结果均回本机。当前五源码＋正常DSO与远端及前轮selected六份逐byte相同；本轮没有改public入口或重跑public两流，沿用前轮通过记录。当前完整17阶段表已打印并保存，阶段数值明确沿10:06历史实测，本轮未重测这些阶段；完整层正常计时是本轮新测。

两个构建、四GPU进程EXITED_OK，无编译/运行失败或timeout；r0细计时筛查失败1项保留，修正1次。5946/6500完整block，保守计15分钟，累计34186.413/36000秒，余额1813.587秒（约30.2分钟），原起点/截止不重置。所有12个owned身份已退出，5090显存0，实例未关机。新增证据低于1GiB。双A10010:43只读见原RoboTwin模型/模拟和巡检正常，不改队列、不重算SR。

四科学权限与production_admitted=false，有限单层诊断不代表在线质量或机器人E2E。

[分量结果](COMPONENTS.md)、[原始汇总](SUMMARY.json)、[当前完整阶段表](STAGE-TABLE.md)、[固定入口](SELECTED.json)、[资源](RESOURCE-AND-STOP-REPORT.json)、[清理](OWNED-CLEANUP.json)、[当前未改byte核验](CURRENT-UNCHANGED.json)。
