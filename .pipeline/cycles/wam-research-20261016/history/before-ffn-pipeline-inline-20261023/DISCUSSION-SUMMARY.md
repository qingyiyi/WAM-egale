# 当前讨论（2026-10-22T22:19:55.013997+08:00）

用户要求先profile当前FFN并讨论原因，没有要求本轮继续改tile或调度。已核对当前B及两机证据，生产源码不变。关键观察：FFN1末组迟约80μs，最后FFN2约92μs计算/18μs写回/1μs提交等待；NCU有29.8MB local和额外整数指令，不能把barrier42.5%当可省墙钟。后续优先讨论尾部形状和寄存器/数据搬运定点对照，不把未测机制说成根因。

用户询问空卡后，明确批准固定1800局160GPU小时/10天新资源；随后纠正要保留一张卡跑算子。最新分工GPU0算子、GPU1全部RoboTwin，不自动借GPU0。新授权已Maintain；旧账本/截止/失败保留。GPU1已启动官方有效初态补齐；原工程组合与优化reset沿显式配置用于各臂，bank缺少新指令时由原GPU T5完整512-token BF16补齐。未启动新训练/TEST，四科学权限false。

[当前FFN](../../experiments/single-layer-async-gemm-20261020/ffn-current-profile-20261022/RESULT.md)；[固定扩大队列](../../experiments/robotwin-pilot-20261020/eval100-20261022/PLAN.md)。
