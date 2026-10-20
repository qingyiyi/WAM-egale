# 最近执行 checkpoint

更新时间：2026-10-20T17:26:40+08:00。

1. 核对当前慢点：真实B14三KV插桩阶段账本和同输入Torch Linear，12完整block/150原语，输出/边界逐位。GEMM主导70.70%–76.08%，不借旧M38占比。
2. 用户明确希望快速GEMM源码嵌入单kernel。固定CUTLASS3.5.1设备mainloop，保留64×128×64和原BF16归约/bias；只改optimized。CPU首次编译类型/缓冲引用错误保留；修正后再构建，原起点/预算不重置。
3. 63block门通过：9签名/23中间量/18有效KV，输入/RNG/tile；重复未测。门通过后两个独立进程各21block，B14仍慢2.54–3.02倍，六trace一kernel。新ready未验。总1497/1500，止损GPU；现场两卡0MiB、8个本次owned根进程回收。
4. RoboTwin首六局原Target全部到终态，5成功、open_microwave自然失败；83policy/6557teacher，实际1520和罚分1500分列。原first6 guard完整清理，后继TF未自动启动。
5. 更新当前HW/源码文档与终态小证据，旧结果、旧FAILED和预算不修改。

[算子结果](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/CUTLASS-RESULT.md)、[资源审计](../../experiments/single-layer-async-gemm-20261020/b14-bottleneck-followup/FINAL-AUDIT.json)、[RoboTwin](../../experiments/robotwin-pilot-20261020/FIRST6-TARGET-RESULT.md)、[前checkpoint](history/device-gemm-followup-start-20261020/EXECUTION.md)。

最新只读checkpoint 2026-10-20T18:33:05+08:00：核验CUTLASS复用边界与原ptxas spill报告，确认默认仍阶段全局同步；保存用户B14≤2倍工程目标。远端两卡无compute进程，RoboTwin首六局5/6结束、owned无残留，TF链未实现齐，旧准备/首六局截止已过。未新增GPU或编译。
