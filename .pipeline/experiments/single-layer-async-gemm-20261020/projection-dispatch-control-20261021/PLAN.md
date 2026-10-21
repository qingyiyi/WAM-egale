# 同计算体CTA调度诊断

- [x] 用户确认与预算登记。
- [x] optimized诊断入口、同一个compiled kernel三模式。
- [x] 三KV、K/V各自逐位与tile覆盖审计。
- [x] 两独立进程、hot及相同原版前序计时；队列清零单列。
- [x] 相同8warp计算体嵌入整层的计时与编译资源比较。
- [x] 原始结果、失败/清理/累计预算、HW与GitHub收口。

数学源来自已检验但整层变慢的256×128/8warp候选；仅诊断，不替换当前选用4warp。普通672CTA也是一个launch；不把实验设计写成收益。GPU0独占，GPU1 RoboTwin继续。

边界：GPU测量完整收口；嵌入阶段有插桩扰动，精确成本归因UNKNOWN，详见RESULT，不把缺口勾作已解决根因。
