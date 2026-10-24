# 三个小投影有序split-K：不采用，恢复上一版

终态 **CLOSED_REJECTED_PERFORMANCE_RESTORED_A**。评估变体PROJECTION-SPLITK；当前实际保留 **SPLITK-M64-FFN2**。本轮没有新增性能收益，FFN2此前收益保留。

## OBSERVED

三个小投影为Self输出、Cross Q、Cross输出（参数索引9/13/19）。只将原96个输出tile各十个K段，拆为960个跨CTA任务；保留8MMA warp、M64×N128/K64、九段K320＋尾K192，原BF16归并和末段bias顺序，原FFN2不动。每投影独立领取/提交状态，总新增1164B全局计数；未启用TMA。

候选三KV真实B14/Q16/block0的26中间量＋输出、前后normal、KV/输入/RNG逐位和状态通过；投影claimed=[960]*3，commit=[3,96]全10；FFN2原计数门不变。CPU有限调度及同步审查通过。它数值正确，但性能不合格。

|候选对照，两独立进程|短KV288|中KV1152|长KV2160|
|---|---:|---:|---:|
|候选相对同场A完整层耗时变化|+42.700%|+40.873%|+38.087%|
|原Torch/候选|0.8063×|1.1549×|1.4783×|
|原Torch/当前保留A|1.1505×|1.6270×|2.0413×|
|Guarded/当前保留A|1.0193×|1.0017×|0.9710×|

全eager、无Graph、同GPU/输入/缓存、四臂交错。两个进程每档均退化；按登记停止，不继续参数搜索。当前A相对guarded短快1.889%、中快0.171%、长慢2.987%；小于1%的差异不宣称稳定优势。

分时完整层插桩诊断中，三个投影候选区间约0.162–0.176ms，当前A约0.0596–0.0620ms，为约2.63–2.94倍；其他主要区间接近。该诊断定位退化集中于改动投影，采用决定仍来自同场无插桩完整层。插桩扰动候选绝对≤0.334%，当前≤0.293%。

候选和A的静态资源均170CTA/每SM1CTA/256线程/100096B shared/255寄存器/784B local，未发现静态占用上限变化。实际动态流量/等待时间尚无硬件计数器证据。

## 解释边界：HYPOTHESIS / UNKNOWN

新组织增加了细粒度领取、跨CTA有序等待、全线程发布和同步。原FFN2每个K2048任务有32次K64迭代；小投影大多数K320任务只有5次、尾段3次，相同M64N128下每个主任务理论算术量仅FFN2的1/6.4。短任务更难摊薄调度成本，这是有源码支持的机制假设，不等于已定量证明atomic、spin或threadfence各自占比。

不能因为静态资源相同宣称运行时所有资源行为相同，也不能说“所有split-K均无效”。Torch480CTA来自48个128×128输出tile×10段；本候选960任务来自96个64×128 tile×10段，且是170个persistent CTA领取，不能混为等价调度。

## RECOMMENDATION

继续使用FFN2 split-K，对三个小投影保留原串行K段。下一步若继续，先拆量任务领取、等待、提交/发布开销；再只试一个更粗任务，例如每任务保留5个K段顺序计算，变192任务，并保持每段原BF16舍入。该建议尚未实现、未登记为新GPU作业。暂不再把更细任务或TMA当作必然收益。

## 回退、当前表与资源

实际optimized五份源码已恢复A，选用独立wam_whole_splitk_cuda。默认与非默认public流、当前阶段重测均通过；旧候选源码、正常/profile二进制、全部结果保留。六份源码/二进制回本机byte一致。当前完整表为[STAGE-TABLE](STAGE-TABLE.md)，17阶段本轮新测；原候选完整表在[PROJECTION表](reports/PROJECTION/STAGE-TABLE.md)。

本轮完整block11469/15000，两个构建和六个GPU进程均EXITED_OK；没有数值失败或timeout，性能候选拒绝1个。GPU worker进程生命周期54.663秒，CUDA active为UNKNOWN。保守生命周期15分钟，累计32386.413/36000秒，余额3613.587秒（1.004小时）。新增约37MiB，小于1GiB。16个owned PID/出生身份均退出，5090显存0，实例未关机。原预算起点/截止保留。

四科学权限及production_admitted均false，单层结果不外推30层、接受路径或机器人E2E。

## 证据

- [固定当前与候选身份](SELECTED.json)
- [候选两进程汇总](reports/PROJECTION/SUMMARY.json)
- [当前重测阶段](results/RETAINED-stages-data/RESULT.json)
- [同步审查](SCHEDULE-AUDIT.md)、[源码/结果复核](INDEPENDENT-REVIEW.md)
- [资源](RESOURCE-AND-STOP-REPORT.json)、[清理](OWNED-CLEANUP.json)、[回传byte核验](RETURNED-BYTE-CHECK.json)
