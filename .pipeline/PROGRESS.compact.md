# WAM 当前短状态

更新2026-10-21T02:34:36.439687+08:00。按[策略](CONTEXT-POLICY.md)与[约束索引](memory/INDEX.md)读取当前Cycle，
不全量sync旧HW。[PLAN](cycles/wam-research-20261016/PLAN.md) /
[完整当前进度](cycles/wam-research-20261016/PROGRESS.md) /
[EXECUTION](cycles/wam-research-20261016/EXECUTION.md)。

- 用户主线：先单层原语/性能对齐，再调度；不能用异步潜在收益掩盖差距。
- M39：活动源码为RMS向量搬运后顺序简易版。九签名/23中间量/18有效KV转移/输入/RNG有限逐位误差0；
  B14短/中/长完整层仍为native的1.47–1.56/1.48/1.34倍，未速度对齐。
  [三KV阶段、全部两进程原值与原因](experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md)。
- 独立stage仍含整层shared/队列，不等于纯算术；C−B不是纯barrier。
  原语差距、统一资源瓶颈有证据，动态spill/HBM/L2/occupancy未知。
  compact2CTA/SM变慢2%–10%已撤回，N256/point8192/强制内联负结果保留；无新ready。
- GPU：最新两卡均无计算进程，owned子进程清理完成；不是研究全部完成。
  本夜5970/6100 block、13572/14000 stage，GPU worker生命周期620.0秒；
  M39累计约1285.8秒/10小时。CPU构建单列，CUDA active未知。
- M40：Target48（41成功7自然失败）、PARENT2000/CAL齐；parent17/18（13成功4失败）齐。
  固定FIT11/12、CAL6/6；缺valid08原PARTIAL补跑待答复，不重试、不缩减分区。
  后继训练/current18/DEV90未启动。[原件与结论](experiments/robotwin-pilot-20261020/OVERNIGHT-RESULT-20261021.md)。
  GPU累计5.619/24小时；旧截止/失败保留。
- LIBERO6000局齐，T/C/P各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。
  曝光/首两S审计/长尾缺口保留，不叫独立TEST。
- engineering_quality_screen=false；independent_test_complete=false；
  noninferiority_established=false；formal_performance_permission=false。
  旧A_off默认、paused72、MLP0、22缺收据、FAILED/retention不变。
- baseline/原tile/LIBERO投机与state/cycle/rules保护。BAAI只~/cxz，AutoDL已归还；
  不改驱动/服务/时钟、不杀他人；GitHub仅main。
- 下一步仍先GELU/table及FFN2阶段对齐；RoboTwin补跑待此前答复。
  10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00中文证据包。
