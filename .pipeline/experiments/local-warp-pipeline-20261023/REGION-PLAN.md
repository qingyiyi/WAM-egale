# T1 局部准备方案

OBSERVED：锚点ffn2_partition W3–6完成128×128/K7168，使用128线程bar2；整个CTA在返回时会合。ffn2_commit原8warp向量提交，bias/table已按列移出循环；第一段BF16 partial在原group完成计数acquire后可读。R2和temb在FFN之前就绪。

选择唯一首版：W0/1/2/7准备当前第二段tile的bias/table以及首8行gate/R2，W3–6保持MMA。最终仍8warp提交，不改第一段发布/96任务/两段K/原舍入。准备区：bias128×FP32 512B，table512B，gate8×128×FP32 4096B，residual8×128×BF16 2048B，合计7168B。未缓存整tile/partial。其余行沿原计算。首8行占满tile6.25%，尾tile8.33%；可隐藏收益有限，必须测净收益。

独立FfnLocalTile将math union与prep并列；位于原TileShared union内。它不与本tile mainloop/result重叠；预计98304+7168=105472B，不超过原FA2占据的TileShared。编译static_assert限制不超过112128B，真实sizeof/occupancy以GPU资源输出确认。前一FFN1返回及发布屏障后才能转作此workspace；commit完成+ffn_publish屏障后才复用下一tile。

B/C使用同一新二进制、同一Params/资源，仅forward的ffn_prepare_overlap bool不同：B先四warp准备，全CTA会合，再MMA；C四warp准备与MMA并行，结束全CTA会合。第一段无准备。A独立旧DSO不含新代码。B比C多出的串行会合属于所测调度成本，控制并非所有指令完全相同。

准备不读取partial；原group acquire、bias+partial2再partial1、BF16、gate乘法、残差加法不变。每次tile重写自己的prep，不使用跨Graph ready标志。所有CTA线程参加同一全CTA屏障；MMA内128线程bar2仅W3–6。

T1其余区间：LN仍4warp Welford与8warp搬运；RMS两4warp完整D行；FA2两4warp合法KV；小投影8warp；FFN1主8/尾4warp。独立数据范围与各屏障见STAGE-T1。

证伪：输出/partial或状态任何不等；新增shared挤出驻留；C/B没有可重现改善；C/A退化或三KV方向混合。静态local增加不自行解释动态退化。轻量软件时间线只在独立诊断副本，正常计时不插桩；最多一次有定位的局部修订。
