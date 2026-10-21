# 当前执行证据（2026-10-21T22:19:31.835566+08:00）

**RoboTwin扩大前先诊断慢因；1800局未启动，新增160 GPU小时未批准。** 旧DEV90保留，各30局T/TF/PF2成功25/24/23；非独立TEST，四权限false。用户已把扩大目标从50改为官方每任务100次，六任务三臂1800局，未扩大到全部50任务。

## 慢因与原Target复现

- [90局成本与完整任务profile](../.pipeline/experiments/robotwin-pilot-20261020/runtime-profile-20261021/RESULT.md)：平均客户端生命周期262.81秒；RPC外60.02%，reset90次合计1.115秒（0.0047%）。18失败局占55.11%时间，不能删除或截断来加速。
- 一条完整成功局profile253.6秒：RPC102.5、动作72.5（内含双臂规划60.5）、动作外观察46.6、setup18.7、其余13.4。命令与未profile一致；约5.36%插桩开销，不作正式计时。
- 可研究未用front相机和重复planner初始化/预热；尚未实施，收益未知。同步函数的39秒包含等待实际工作，不能直接当可消除开销。
- [原canonical Target两局复现](../.pipeline/experiments/robotwin-pilot-20261020/microwave-reproduction-audit-20261021/RESULT.md)：valid14失败/valid16成功，完整命令及保存初态/首两policy/RNG与旧Target相同；11份官方固定源码相符。只证明覆盖两例，非论文全部复现，不增加独立质量N。
- 计数语义修正：[派生表](../.pipeline/experiments/robotwin-pilot-20261020/runtime-profile-20261021/DEV90-COMMAND-SEMANTICS.json)区分提交12224/13376/13408与环境实际推进11808/12927/12958；旧actual_commands为提交调用数，旧记录/罚分不覆盖。
- [100次具体方案](../.pipeline/experiments/robotwin-pilot-20261020/microwave-reproduction-audit-20261021/EVAL100-PLAN.md)待新吞吐/资源确认。原M40累计51483.683/86400秒，原10/22 01:26截止保留，owned退出。

## 算子精简与后续

- [结果](../.pipeline/experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)：Python直接绑定原参数，移除两个死函数，普通返回不组装debug中间字典/资源统计。设备数学、tile、KV、同步不变，baseline/投机/原tile不动。
- 9真实签名及9组普通/debug有限逐位通过；两进程三KV完整单层Graph变化−0.132%～+0.103%，无稳定退化，无新增设备加速主张。
- 当前源043c7759、DSO2a8a250e，两机一致；累计2538.680/36000秒，owned退出。[源码入口](../算子研究/wam_ops/optimized/whole_layer.py)、[CUDA整层](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。
- 原小投影1.88%–2.08%收益保留；Self-Q、FFN原语仍待对齐，ready-DAG未实现。不能把此前计时拼成当前Torch倍数。

原paused72/retention/FAILED与review/RESULTS.json保持历史身份。LIBERO6000局是三方法各四套×500，一轮，不是每方法三seed重复。原tensor/DSO/profile留服务器，公开证据缺失范围见EXPORT-NOTES。前派生快照commit `dbf02563f3e4729541483ab069c5496e22b31bae`。
