# 连续TMA小投影分量诊断

用户明确同意先分量计时，再决定后处理warp交接。本轮只派生诊断副本；当前optimized源/独立normal二进制及采用身份不变。

- [x] PREP：冻结原TMA候选和当前A，核对预算/资源。
- [x] INSTRUMENT：同一诊断二进制提供OFF/COARSE/FINE；保留所有数学和同步，只加thread0 globaltimer＋独立shared统计，结束时输出每投影96tile记录。
- [x] EXACT：三个真实B14/Q16 block0输入，完整中间量/输出/输入/KV/RNG及原FFN/TMA计数核验。
- [x] MEASURE：两个独立进程，原TMA正常、诊断OFF/COARSE/FINE交错；测模式扰动、静态资源和观察者区间；不把相加的CTA工时当整层时间。
- [x] CLOSE：分量表与判断、当前完整17阶段表（未改源码，阶段沿10:08历史值明确标注）、源码/DSO/证据回传及owned清理。

预算35分钟包含准备/编译/等待/失败/清理，原累计33286.4125/36000秒不增；CPU4、5090单worker、完整block6500、新存储1GiB，预留300秒清理。最多一次有依据修复，不新增优化候选或warp分工，不改A100/投机/baseline/保护state。

计时边界：wait仅leader轮询等待；ready同步、operand/shared→register＋MMA、reader释放同步、TMA发起、accumulator落shared＋同步、BF16/bias/写回、epilogue同步分别记录。shared→register与MMA交错，不伪造纯TC或带宽时间；CTA同步含观察者到达偏差。COARSE只在数值段边界记时，用于检验FINE扰动。每tile总时长、分量和未归属间隙明确分开。

投入筛查：诊断OFF相对原TMA正常偏差≤5%、COARSE相对正常≤5%、FINE≤10%才谨慎给定量分量。超出时保留原结果，不能把百分比写为原实现真实stall比例；可报告方向或UNKNOWN，不无限修计时器。没有硬件计数器，不能报TensorCore/DRAM利用率。

不自动实现后处理warp交接；仅在证据支持时给下一步方案。四科学权限和production_admitted保持false。

GPU执行前补充观察者尺度核验：即使完整层FINE扰动≤10%，还需逐投影FINE/COARSE tile_total的均值偏差≤10%才能将细分量作定量参考，避免完整层其他阶段稀释局部插桩开销。超出则细分百分比标UNKNOWN；COARSE段级观察仍单独审查。

r0两进程FINE局部扰动约14%超过10%，不作精确stall归因；已保留。使用唯一一次r1诊断修正：相邻边界共享时间戳，减半相邻读时钟，不改数学/同步。新每进程10轮×4、5预热、4trace，总调用仍≤6500；旧起点/截止/筛查线不变。

终态：r0 FINE扰动超限保留，r1唯一修正通过工程扰动筛查，全部数学/状态通过。仅诊断完成，当前采用代码不变，未启动输出交接。
