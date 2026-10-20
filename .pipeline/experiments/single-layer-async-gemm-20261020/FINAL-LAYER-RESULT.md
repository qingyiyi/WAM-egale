# 最终单层结果：有限精度对齐，性能未对齐

OBSERVED：默认阶段路径与FFN行组ready路径均在完整单层的一次cooperative kernel内计算。最终同一二进制、9真实输入×2路径=18格，最终输出及全部可配对的已保存DAG边界对原Torch/native逐位一致；每路径6个起点×update1→2→0，共18次合法有效KV转移直接对原Torch通过。输入/RNG/重复/tile审计均通过。只有9种底层真实输入，18格不是18个独立机器人样本。

最终去掉整份Params设备端复制，并以const __grid_constant__引用只读launch描述符。kernel stack1200→408B、localSizeBytes2664→1880；仍255 registers、121376B shared、108CTA×256线程、理论1CTA/SM。静态编译资源变化不等于已测到spill流量或根因占比。

|真实输入|native ms|默认阶段 ms|行组ready ms|默认/native耗时|ready额外耗时|
|---|---:|---:|---:|---:|---:|
|short-B2-action|1.4403|4.1523|4.2061|2.883×|1.29%|
|middle-B2-action|1.5500|4.3686|4.4497|2.818×|1.86%|
|long-B2-action|1.6888|4.5244|4.5796|2.679×|1.22%|
|short-B14-action|2.5778|12.9445|13.2950|5.022×|2.71%|
|middle-B14-action|2.4073|10.8687|11.1605|4.515×|2.68%|
|long-B14-action|2.8418|11.2424|11.5199|3.956×|2.47%|
|short-B2-video|1.4968|5.4706|5.7970|3.655×|5.97%|
|middle-B2-video|1.5839|5.6070|5.9433|3.540×|6.00%|
|long-B2-video|1.7110|5.7935|6.1307|3.386×|5.82%|

每格两独立进程各2样本，全部4样本合并中位数；完整原始108样本/host时间、每进程数据均在final-followup。样本量只适合本轮工程筛选，不宣称普适性能或统计非劣。18条ready trace及9条默认trace均只有1个compute kernel，profiler样本不混入计时。输入清单与有限数值门一致。

启动前GPU0空、每case单NVML context；host PID与container PID映射未独立证明，原始两组ID保留。共享文件系统/其他CPU任务仍是机器条件；这里是uncached eager单层计时，不是最强cached/Graph、完整verifier、chunk或机器人E2E。没有从block虚构有效提交g。

参数修正参考[NVIDIA CUDA12.4官方说明](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_C_Programming_Guide.pdf)的__grid_constant__；它避免取地址时的逐线程参数副本。此次同时删除显式Params拷贝，且旧/新二进制未做同一时段双进程受控消融，因此不把跨阶段几percent变化全部归因于这一个限定符。

RECOMMENDATION：默认tile_pipeline=False。保留ready研究入口及负结果，不集成机器人，不扩大ready队列。默认仍慢native2.68–5.02倍，优先解决大GEMM/归一化与资源效率；H3/H4未成立。B6/3状态未准入，cache/Graph完整矩阵未完成；没有用更弱baseline包装新增收益。

O1/O2/O3/O4分别261/300、390/400、432/500、297/300完整block，共1380/1500；准备失败、实施无效与所有负结果保留。资源去重及退出见RESOURCE-JOBS-DEDUPLICATED.json和FOLLOWUP-STOP-REPORT.md。旧M39登记FAILED与四质量false不变。
