# M34资源与止损

终态：有界单层原型和诊断完成；原精度、性能均NO-GO。CUDA主源码固定df8b386…，新增布局仅为独立诊断副本。登记600完整block调用已全部使用，不再启动此轮GPU作业。

| 口径 | 实际值 |
|---|---:|
|远端首批3个GPU worker生命周期合计|66.555秒|
|远端profile/布局阶段GPU预留生命周期（含CPU编译）|86.720秒|
|其中2个GPU worker生命周期合计|28.622秒|
|本机3个作业生命周期合计|47.590秒|
|其中首个依赖失败、0block|6.255秒|
|总CUDA active秒|UNKNOWN|
|完整30层/机器人episode|0/0|

不同生命周期边界没有相加伪装成CUDA active。CPU等空卡的时间在queue原时间戳中保留；原7小时父上限的未用时间不自动转授新作业。0timeout。两次CPU编译失败、一次本机依赖失败及全部数值失败保留。旧M32/M33及历史22缺失收据没有补造。

本机88保护项最终核对不变。远端首批86已部署项不变，原88项检查因state/rules未迁入而为false，原Missing不修造；阶段诊断未改主源码/DSO。所有本轮owned root已退出或reap，01:44:39北京GPU0 context为空；GPU1仍是另一登记的Long计时，不是算子残留。

止损方向：不继续布局/寄存器/grid搜索，不接入生产，不扩30层，不放宽旧精度门。下一步先评估设备GEMM/attention与固定Torch数学契约能否一致，再考虑带成熟流水的tile原语；不能把已有prompt缓存作为新增调度贡献。L2/DRAM/bank-conflict/achieved occupancy仍UNKNOWN。
