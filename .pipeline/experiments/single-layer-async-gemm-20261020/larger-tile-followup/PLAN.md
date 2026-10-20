# M39更大tile与写回接续

用户2026-10-20最新明确“算子侧请你继续实现，然后tile可以切大一些；robotwin继续关注”。本次是同M39的新有界接续，旧1497/1500终态不改、原20:20:28北京硬截止不延长，不借旧剩余3次冒充完整验证。起点：2026-10-20T19:08:53+08:00。

目标：单层一个kernel，B14短中长均争取≤同轮Torch两倍，输出/中间量/有效KV/RNG不放宽。只改optimized；baseline、投机、原tile和保护HW不动。

- [ ] A：GEMM协作成对写回、设备mainloop内联、point/cache任务128→2048元素。GEMM输出tile仍64×128。
- [ ] B：保留A，额外将大M条件投影（M≥1024）tile改128×128；action M224保留64×128以保持CTA任务数。最多这两种，不做网格。
- [ ] 每版9真实签名72完整block门：初始Torch/native/new＋重复、18合法KV转移、23配对中间量、输入/RNG/tile审计。失败不计时。
- [ ] 每版通过后两个独立进程各57完整block（B14三KV，每格2warm＋8轮交错native/new＋1profiler），普通计时与profiler分离，保持原uncached eager边界。合计计划372，硬上限420完整block，0机器人episode/teacher/新原语扫描。
- [ ] 以精度合格及三KV最差耗时比选当前研究实现；大tile若更慢保留负结果并恢复较快版，不以size作为成功标准。不能把该工程选择称独立测试或论文结论。

GPU0单worker，CPU4/5两核；GPU worker生命周期合计≤900秒，含构建/等待/审查总墙钟≤3600秒，且最迟原20:20:28北京清理，预留60秒。失败/timeout/准备计入，源修复单列attempt，不自动重复相同失败作业。新增证据≤512MiB，共享真实fixture不复制。原始旧FAILED与四质量false保留。

启动记录输入路径、源码清单、构建宏、GPU UUID/外部负载及PID+出生时间。每次完整调用前计数，guard预算不足停止；终态明确与owned清理核验。RoboTwin只读核首六局状态、未实现依赖和资源，不重启过期链。
