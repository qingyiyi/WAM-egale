# O2：数值通过，原语效率仍未达到 native

当前 optimized 采用 Norm 向量 IO、M≥128 的 64×128×64 GEMM，M32 保留 64×64×32。最终独立进程九点及18合法KV序列重新通过；两轮同名扩展初测无效，不作为优化消融。

|真实输入|native ms|optimized ms|optimized/native耗时|
|---|---:|---:|---:|
|short-B2-action|1.4065|4.3105|3.065×|
|middle-B2-action|1.5498|4.4658|2.882×|
|long-B2-action|1.5549|4.0325|2.593×|
|short-B14-action|2.2639|11.6390|5.141×|
|middle-B14-action|2.4030|11.1129|4.625×|
|long-B14-action|2.8379|11.4922|4.050×|
|short-B2-video|1.4938|5.6463|3.780×|
|middle-B2-video|1.5849|5.7845|3.650×|
|long-B2-video|1.7116|5.9565|3.480×|

每格两独立进程各6样本，报告全部12样本合并中位数；216计时样本、18单kernel trace，状态准备/插桩在计时外。输入与精度门为同一九fixture清单。此处不是最强 cached/Graph 基线，也不是 chunk/verifier/机器人 E2E。

资源：108 CTA×256线程、255 registers/thread、121376B shared、2544B localSizeBytes，理论1 CTA/SM。NCU/实测occupancy/流量仍UNKNOWN。GPU0启动前空、每case仅见1个NVML进程；host/container PID映射未独立证明，不能伪称该映射通过。

O2累计390/400完整block：profile12、实施身份无效36、有限门72、最终270（含warm/trace）；两次PID命名空间预检各0调用，失败/等待/清理仍记生命周期。全部失败保留，四false不改。

工程结论：仍比成熟native显著慢，不能生产替换或宣称H3。接续已授权O3的最小FFN行组就绪交接用于验证机制；不能预设其会弥补原语差距。完整强baseline及1/3/7×cache矩阵仍待前置，不从这里制造有效提交g。
