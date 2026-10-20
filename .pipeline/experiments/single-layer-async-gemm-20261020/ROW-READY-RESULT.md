# O3：单层内FFN行组就绪交接，机制成立、速度NO-GO

OBSERVED：在当前单个cooperative kernel内实现FFN1/GELU→FFN2就绪调度。完整64行组的所有F14336列tile都发布后，该行组才允许FFN2按原K顺序归约。其他行组的producer可同时继续；未改变self/cross FA2数学、KV或BF16边界。warp0调度，loader/compute/store职责保留。

两版完整门各72调用，均九点输出及18合法KV更新对原Torch逐位一致，可配对的已保存DAG边界对native一致；重复、输入、RNG、tile审计通过。第一版无界ticket导致上万次无效领取，原计数保留；第二版bounded-CAS约束每组112/24合法生产/消费（小M为224/48），release/acquire与每调用重置保证可见性。

B14/video四行组的96个FFN2 tile中，实际记录72个领取时尚未完成全部448个producer；B2 action单行组为0。该计数只证明调度交接顺序，不能当成GPU重叠时长或性能收益。其余子图仍阶段grid barrier，不能称整层无阶段同步。

|输入|native ms|同DSO阶段路径 ms|行组ready ms|ready耗时增加|
|---|---:|---:|---:|---:|
|short-B2-action|1.4085|4.3715|4.4498|1.79%|
|middle-B2-action|1.5528|4.5698|4.6503|1.76%|
|long-B2-action|1.5364|4.2354|4.3007|1.54%|
|short-B14-action|2.3577|12.5723|12.8857|2.49%|
|middle-B14-action|2.4053|11.6029|11.8964|2.53%|
|long-B14-action|2.8405|11.9722|12.2420|2.25%|
|short-B2-video|1.4966|5.7831|6.0915|5.33%|
|middle-B2-video|1.5910|5.9238|6.2464|5.44%|
|long-B2-video|1.7106|6.0807|6.3912|5.11%|

每格两独立进程各4次计时，全部8样本合并中位数；216样本、18单kernel trace。两个进程每cell的ready/static中位数比均大于1，非仅合并统计的反转。启动前空GPU、运行每case单context；NVML host/container PID映射限制及两次0调用入口失败保留。

HYPOTHESIS：新队列/轮询、每tile GELU交接及调度顺序的成本超过省下的阶段等待；这些因素未被独立拆分，不把慢因全部归于某一个atomic。编译记录显示较大local stack，需区分参数副本与spill；NCU stall/流量仍UNKNOWN。

RECOMMENDATION：保持tile_pipeline=False为默认，保留研究路径而不生产集成。O3共432/500block，未无界搜索。后继在原O4尾部预算内只处理编译器参数复制并最终核对引擎；完整1/3/7×cache强baseline矩阵不能因此宣称已完成，B6仍不准入。新尾部源码结果与本次固定版本分开。
