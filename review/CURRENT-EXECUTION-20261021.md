# 10/21：先对齐，不先做调度

固定本文件所在commit审阅；历史paused72结果保持原身份，不能用本页覆盖原失败或四项false。

当前简易整层persistent已恢复较快顺序版，仍逐阶段grid.sync，没有新增ready调度。
九真实签名输出、23中间量、18有效KV转移、输入/RNG有限逐位误差0；完整B14层尚未速度对齐：
短/中/长分别约native的1.47–1.56/1.48/1.34倍。计uncached/eager单层，不是strong cached/Graph、30层、chunk或机器人E2E。

[完整两进程数据/最新阶段/原因](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESULT.md) /
[源身份与状态JSON](CURRENT-EXECUTION-20261021.json) /
[资源和全部失败](../.pipeline/experiments/single-layer-async-gemm-20261020/stage-alignment/overnight/RESOURCE-AND-STOP.md)。

主要已知：独立阶段FFN2约1.41、GELU3.40、table2.58倍；统一kernel256线程/255reg/98848B shared/1CTA每SM。
独立阶段仍保留cooperative队列、同步和统一shared预留，因此不是纯算术实验。
compact双驻留方案虽然attention改善，小投影/FFN2退化，完整层慢2%–10%，已撤回。
N256/point8192/强制内联也未选；RMS向量搬运稳定改善1.1%–1.7%已选。
C−B不能当纯barrier，动态spill/HBM/L2/occupancy未知。没有证明单kernel理论上无法对齐。

代码从[whole_layer.py](../算子研究/wam_ops/optimized/whole_layer.py)、
[整层顺序](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、
[GEMM](../算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh)、
[FA2](../算子研究/wam_ops/optimized/csrc/fa2_stage.cuh)、
[代码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)读起。

RoboTwin：Target48采集41成功7失败；PARENT2000完成；parent采集17/18齐、13成功4自然失败，
固定FIT11/12、CAL6/6。原valid08在0动作接口错误PARTIAL，修复后加强policy门通过；
补跑待用户答复，未自动重试，后继训练/current18/DEV90未启动。
[RoboTwin结果](../.pipeline/experiments/robotwin-pilot-20261020/OVERNIGHT-RESULT-20261021.md)。
两卡最新终态均无GPU计算进程，科学研究仍未完成。

网页版审阅请求：请依据本commit的源代码和同输入A/B/C测量，先判断剩余性能差距。
优先检查GELU/table向量化、FFN2分段BF16 epilogue、独立stage的统一shared预留和整层资源约束。
每条建议写清源码依据、预计影响哪个实测阶段、最小可证伪实验，以及能否保持原BF16逐位契约；
不要跳到跨阶段ready调度，不把普通batch/cache、调用减少或静态occupancy当新贡献/实测E2E。
保留UNKNOWN，不建议放宽门限制造PASS。

四权限全部false，旧默认/retention/FAILED不改；baseline、LIBERO投机与原tile没有修改。
公开仓库不含原始PT/KV/权重/DSO，不能声称仅凭此仓库独立重算了全部tensor。
