# 当前输出槽交接诊断结果

评估变体：RESULT-EMPTY-OBSERVER（仅观察副本）；当前实际保留：WARP-SPECIALIZED-4MMA。三个小投影仍四MMA／一TMA／三epilogue、64×128输出、48个K64、十段BF16有序合并。没有增加warp或TMA store。

## 已观察

三份真实block0/B14/Q16短中长KV288/1152/2160，246次完整block调用。所有捕获中间量、输出、有效KV、输入和RNG有限核验通过。正常与观察副本资源同为170CTA、256线程、100352B shared、255寄存器/线程、784B静态local；不能据静态相同推断动态spill相同。

|KV|投影|empty每次均值 μs|P50/P90 μs|九次empty合计均值 μs/warp|store＋发布均值 μs|epilogue计算＋释放均值 μs|epilogue等full均值 μs|
|---|---|---:|---|---:|---:|---:|---:|
|short|self_out|0.0413|0.032/0.064|0.3714|0.1923|1.4819|2.8020|
|short|cross_q|0.0414|0.032/0.064|0.3729|0.1829|1.3932|2.9497|
|short|cross_out|0.0412|0.032/0.064|0.3708|0.1864|1.4447|2.9239|
|middle|self_out|0.0414|0.032/0.064|0.3724|0.1859|1.4640|2.8542|
|middle|cross_q|0.0412|0.032/0.064|0.3707|0.1845|1.3890|2.9765|
|middle|cross_out|0.0409|0.032/0.064|0.3684|0.1826|1.4424|2.9151|
|long|self_out|0.0413|0.032/0.064|0.3715|0.1891|1.4851|2.8300|
|long|cross_q|0.0414|0.032/0.064|0.3724|0.1904|1.3874|3.0034|
|long|cross_out|0.0412|0.032/0.064|0.3706|0.1821|1.4586|2.9052|

初段不执行empty等待，读钟控制P50同为0.032μs。九次empty原始区间合计约0.37μs，后九段单MMA warp包络约37.3–37.6μs，原始比约0.99%–1.00%。该比值含计时/屏障指令开销，是有插桩单warp观察量，不是整层硬件stall百分比，不据此声称能获得1%的完整层改善。

### 完整层扰动（本轮诊断口径）

|KV|当前正常 ms|观察副本OFF ms|ON ms|OFF/正常变化|ON/OFF变化|
|---|---:|---:|---:|---:|---:|
|short|0.749648|0.757200|0.755520|+1.007%|-0.222%|
|middle|0.813648|0.819792|0.822576|+0.755%|+0.340%|
|long|0.894128|0.902480|0.899520|+0.934%|-0.328%|

同进程、每臂10次交错单调用CUDA事件；用于本轮插桩筛查。不同于上一轮两进程多调用正式块比较，不替换历史Torch加速比。ON/OFF小幅负值属于本次波动，不视为优化收益。

## TMA边界与源码

当前 whole_layer.cu:328–334 中，CUTLASS IteratorC.store(accum)由MMA warp把FP32寄存器结果存入shared，再到达result_full。whole_layer.cu:202–245 中，epilogue读取shared、与前段BF16结果进行FP32加法、最后加bias并BF16舍入，再以B4向量写global；338–345释放result_empty。观察副本保留这些计算和屏障。

NVIDIA cp.async.bulk／cp.async.bulk.tensor的搬运地址空间不包含寄存器源，故不能直接将这一步寄存器→shared交给TMA。shared→global可以考虑TMA store，但要先有符合布局的BF16 shared输出、generic→async proxy可见性和拷贝完成/槽复用协议。它不替代本项目的FP32加法及BF16舍入次序。新增shared开销和实际延迟均未测，本轮未实现。

官方依据：https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk ；stmatrix是warp协作寄存器存shared指令，不是TMA，其类型/布局限制也不能直接忽略：https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-store-instruction-stmatrix

## 假设、未知和建议

- 观察支持：在本轮三份输入中三个epilogue warp能跟上，未见持续输出槽背压。增加epilogue warp的优先级下调；没有测试12/16 warp，不能断言其所有路径均无收益。
- 未知：未插桩路径精确硬件stall、逐tile计时扰动、其他block/在线输入表现。完整层扰动通过不等于局部扰动为零。
- 建议：保留当前8 warp；若继续，应先定位当前四MMA版本的输入等待、shared→register/操作数转换与MMA主循环。旧八MMA诊断不能直接充当当前分量结果。此为建议，不自动启动新作业。

## 资源与证据

本次生命周期含回传145.891秒，保守计180秒；累计35746.413/36000秒，余253.587秒。一次构建、一次GPU worker，全部正常退出；两条CPU核对命令失败已保留，最终七项源码/二进制byte比较通过，未做SHA。5090无compute context，实例仍开机。四科学权限及production_admitted保持false。

`RESULT.json`、`short/middle/long-timestamps.pt`、`STATUS.json`、`RETURN-AND-CLEANUP.json`、`NORMAL-UNCHANGED-r2.json`、`RESOURCE-AND-STOP-REPORT.json`；源码`whole_layer_observer.cu`，独立二进制`build/wam_whole_result_empty_cuda.cpython-312-x86_64-linux-gnu.so`；当前完整阶段快照`STAGE-TABLE.md`。
