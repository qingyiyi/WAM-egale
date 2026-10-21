# 当前单层仍慢的原因：阶段时间线与本机NCU

2026-10-21。**本轮诊断完成，GPU已释放；没有新增数学/调度候选。** 当前完整层差距仍以此前两独立进程未插桩数据为准：B14短KV慢29.01%–29.32%，中慢24.57%–24.68%，长慢14.33%–14.49%。NCU时长不替换这张性能表。

## OBSERVED：本机实际硬件计数

源df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a，真实middle/B14/action/layer0输入。完整自有层及三对native库GEMM/自有独立阶段，共7个NCU范围；每个范围输出有限逐位一致，有效KV核对通过。

本机A100、SM80、Torch2.9/cu126，复用已核对的同源CUDA12.4二进制；CPU2。启动前没有compute/graphics进程、0MiB显存，但NVML利用率残留25%，完整记录，不称正式独占性能证据。`--clock-control none --cache-control none --replay-mode kernel`。NCU明确提示缓存和时钟未控制，结果是本次hot阶段诊断快照；重放次数未单独取得，不把7范围写成7次实际硬件执行。

|阶段|native / 自有 Tensor管线活跃度|native / 自有活跃warp比例|自有/native DRAM字节|自有/native L2 sector请求|native / 自有 local路径读写|
|---|---:|---:|---:|---:|---:|
|Cross K|88.50% / 70.29%|12.49% / 12.50%|1.691×|1.402×|0 / 27.42MiB|
|FFN1|78.75% / 32.90%|12.49% / 12.49%|2.083×|1.515×|0 / 27.42MiB|
|FFN2|81.47% / 52.85%|6.23% / 12.49%|1.194×|1.082×|0 / 24.68MiB|

三个自有阶段的barrier-stall分别46.64%/50.90%/50.86%，native为6.26%/4.54%/0.15%。这是按warp统计的issue-stall指标，**不是墙钟损失比例，也不是grid.sync专用计数**；部分warp在等其他warp执行必要MMA，不能声称“去同步就能快一倍”。

自有整层：Tensor管线活跃40.75%，活跃warp12.50%，eligible warps0.19/active cycle；local-path load181758976B、store162195648B，合计328.02MiB；DRAM1586096256B。local路径字节与DRAM字节不能相加或互相替代，不把全部local访问称HBM流量。没有同轮完整native层硬件counter，不能声称整层DRAM比native增加多少。

NCU报告自有shared为99872B/block，源码动态申请为98848B，两者差1024B；记录不同统计边界，未单独归因这1024B。

原件：[NCU.csv](local-ncu-exclusive/NCU.csv)、[解析JSON](local-ncu-exclusive/NCU-SUMMARY.json)、[数值检查](local-ncu-exclusive/NCU-WORKER-RESULT.json)、[终态](local-ncu-exclusive/STATUS.json)。独立hot阶段和完整层缓存前序不同；不把独立阶段流量相加冒充完整层。

## 机制判断

1. **仅用“occupancy低”解释不足。** Cross K和FFN1的native与自有活跃warp比例几乎相同，自有Tensor管线活跃度仍更低；FFN2的native活跃warp比例更低却更快。资源上限是约束，不能当唯一根因。
2. **GEMM工作映射与数据复用仍未对齐。** native Cross K实际kernel为128x256、grid(24,28,1)、256线程，FFN1为256x128、grid(56,2,1)、256线程；自有源码为128x128输出tile，只有warp3–6执行CUTLASS主循环，其余warp主要等写回。结合真实M/N，native grid与Cross M256×N128、FFN1 M128×N256较大输出tile相符（这是布局推断，不声称已反编译cuBLAS全部tile算法）。当前自有FFN1前216个大tile再加16尾tile，库为112个CTA；不能把复用同一CUTLASS MMA核心理解为复制了库的完整tile/warp/epilogue组织。
3. **local状态存取现在有动态证据。** 同源DSO的反汇编在context_linear、action_ffn1_tail、action_ffn2_parallel入口有大量STL、末尾有LDL恢复。NCU证明local路径确有实际访问，原版这三个GEMM为零。函数调用保存/恢复是已见机制；仍不能把全部local请求都叫寄存器分配spill，也没有量出它独占多少微秒。[SASS摘录](local-ncu-exclusive/SASS-LOCAL-EXCERPTS.json)。不要盲目去掉noinline：此前改变函数组织已使整层改善9.46%–10.87%，那组控制与负结果保留。
4. **阶段尾部空位存在，但不足以直接解释所有差。** 当前三KV时间线里屏障最后到齐后释放中位约1µs；Cross K RMS/GELU结束到达跨度只有3–4µs，而计算段约129/42µs。FFN1尾16任务、Cross K/V尾部及长KV attention有明显早到等待，适合之后研究ready补位，不能把所有等待当可节省延迟。

## 先做什么

- 第一优先：对齐GEMM阶段的有效warp分工与较大输出tile，重点Cross K/V与FFN1；同时控制调用边界live state和local保存。沿原BF16舍入/归约，不用减精度制造对齐，不先加入跨阶段DAG。
- 简单并行工作：RMS/GELU在当前执行器中比较静态分派/较大任务粒度，减少每两行或每2048元素的队列及CTA同步；依据完整层收益决定保留，不能仅凭独立核快。
- 原语明确后：用无依赖Cross K/V填实测尾部空位。先明确依赖与workspace生命周期，再替换grid barrier；当前尚未实现ready。

本轮0新优化候选，不宣称上述改法已实现或能回收某个固定百分比。小算子动态计数未单独捕获、各项因果时间贡献仍UNKNOWN。详细三KV阶段/等待及边界见[TIMELINE-RESULT.md](TIMELINE-RESULT.md)。

## 失败、资源与清理

远端NCU权限拒绝；首timeline native完整Graph捕获失败；本机首NCU绝对DSO依赖失败；资源admission受他人作业阻塞，均保留。修复只作用于测量wrapper/隔离二进制依赖字符串，无活动数学代码变化。一次CPU手动admission因剩余517.5s不足原520s预留未启动；缩短同次采集到420s+30s清理，原14:40:10截止不变。

最终NCU worker自然完成，24.244760秒，退出码0，root已reaped/身份消失，本机GPU进程为空。最终完整账本见[RESOURCE-AND-STOP-REPORT.json](RESOURCE-AND-STOP-REPORT.json)。四科学权限false、baseline/原tile/投机生产及state/cycle/rules保持。
