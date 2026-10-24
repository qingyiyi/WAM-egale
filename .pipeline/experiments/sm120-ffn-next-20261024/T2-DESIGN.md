# PAIR2048x2：唯一局部设计与进度证明

基线为独立A，候选只修改FFN2 owner与partial交接。ACT仍由原FFN1完整归约、GELU后写global，24个输出列tile共享；不重复计算FFN1，不变M/N/K_step、不改投影/attention/归一化/数学边界。

|对象|生产者/消费者/owner|位置/生命周期|
|---|---|---|
|ACT|原FFN1；所有对应行FFN2列tile|global；原ready发布后只读|
|本段P_s|本owner的W3–6 MMA；全部8W epilogue|寄存器→compute union.result；消费后才复用mainloop|
|组内R0/R2/R4|同owner全部8W；同owner下一数值段|独立16384B shared running；每段明确BF16 RN；CTA同步后消费|
|组边界R1/R3/R5|当前组全部8W；后组owner|global F2；所有写线程fence后leader release，后组acquire并CTA同步|
|最终R6/OUT|最后s6 owner；层后继|原bias/门控残差顺序；正常不写最终F2，debug输出R6；最终出口grid.sync|

四组为[0,1]/[2,3]/[4,5]/[6]。每段仍清零MMA acc，32次K64生成P_s；初段按原顺序合并前序global BF16（s0不读），显式舍入至shared；次段仍独立MMA，然后合并shared BF16、显式舍入，最终才写global/发布。没有把BF16前序当作下一段MMA初值。

每输出tile从七次owner任务变四次，group-major领取，总384 owner任务、672个段MMA、七次BF16舍入。新debug schema明确为[4,7,96,384]；最终96输出tile的commit仍7，行队列计数为96而非旧168，旧审计器不修改。旧字段保留实际值，新验证器显式检查新schema；不把新任务数伪装成旧协议通过。

sizeof proof：FFN2 compute union73728B，shared running16384B，合计90112B，小于原TileShared98304B。union中其他阶段workspace不同时活跃；本段mainloop和本段FP32 result继续别名，跨段BF16与两者不别名。整层shared额度和CTA数必须编译/现场复核，不能只凭推导声称不变。寄存器和静态local也复核；动态spill无计数则UNKNOWN。

进度：所有cooperative CTA驻留，每CTA进入消费前已完成自身FFN1。每行queue group-major，领取后只等待更小group的同输出tile，所等前驱一定已领取，依赖严格下降，不存在等待尚未调度生产者的环。组内barrier由全部八warp参与；MMA内部仍由原四warp参与128-thread barrier。没有统一block DAG、跨CTA shared、cluster或新warp比例。

normal DSO没有数值trace大tensor或额外global导出。独立profile DSO在debug时导出七段P_s和R_s，与T1实际相同fixture的原始trace逐段比较；正常profile关闭这些写回。正常整层计时只用未插桩A/B，不用profile决定采用。

当前公平执行为eager/no Graph；本轮不新增Graph性能模式，Graph正确性能力不外推。原B2/fallback数学分支保留，当前研究仅B14三份block0实际输入；不把有限验证写成全部模型、在线接受或机器人质量通过。
