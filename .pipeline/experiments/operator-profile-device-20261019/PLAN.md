# 强基线profile与设备原语：已授权1/2

- [x] 按NEXT-PLAN第1/2阶段登记；固定原科学参考与独立增量。
- [x] 冻结baseline，核对同shape prompt投影cache/静态KV/Graph/shared-prefix各路径兼容性；固定可用强对照，未知标明。
- [x] 六个真实B2/B14短中长block，同输入逐位/状态检查，时间线与有限GPU计数分离；只加载block0权重。
- [x] 按强路径热点选择一个子图，先验证device-side原语；严格原舍入/激活/有效KV，不做调度器。
- [x] 原语若数值或性能不合格，最多两次有依据纠错后保留失败并止损。
- [x] 汇总全部数据、运行终态/预算/owned清理，更新HW与main，并按源码讲解。

预算和原始授权见REGISTRATION。CPU≤2，总墙钟12小时含等待；GPU生命周期≤120分钟（profile≤30、原语≤90），完整block≤6000、原语调用≤20000，新增存储≤1GiB。每次GPU作业前检查自然空闲/固定UUID，检测外来进程则停止计时并保留PARTIAL；不杀他人、不改时钟。空卡NVML利用率异常持续时正式独占性能仍UNKNOWN，机制profile可在无可见外来进程时进行并明确标注。

实现只放算子研究/optimized及共享tools，baseline、投机源码/配置/运行依赖、原tile和保护state/cycle/rules不改。D0或profile失败不降低精度、不借旧预算；本轮不是P_seq/P_sched、不是30层fusion、不是新机器人闭环或正式投机性能。

终态：强基线/profile完成；完整device FFN拒绝，FFN1+GELU收缩组件六真实block×两进程过，详见RESULT.md。GPU部分已停止、自有退出；代码/HW/结果已发布main；源码讲解随本次回传。
