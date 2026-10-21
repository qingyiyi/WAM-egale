# 当前执行：同源码嵌入控制与FFN2小幅收益

2026-10-21北京。本轮证实当前同源、同编译callee、同前序及调用点边界下，独立/整层阶段耗时差约±1.4%。前轮Cross函数组织改变已缓解旧的大额集成惩罚；不能拼接旧独立585–594µs与新整层数字，也不能把剩余都归给全局同步或静态local。

唯一新候选是专用FFN2阶段M128、48输出tile×2个K7168分段=96任务并行；保留BF16舍入与归并顺序。9真实签名/23中间量/18有效KV与输入/RNG有限逐位通过，max_abs=0；一个kernel，未ready。
两独立进程、三KV完整层改善0.36%–0.65%；短/中/长约2.745/3.007/3.262ms，仍比native慢约29.6%–29.8% /25.1%–25.2% /14.7%–14.9%。仅uncached/eager单层，不是强cached/Graph或机器人E2E。小幅收益保留，旧generic linear并行K负结果保留。

[完整报告](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md)、[原因与未知](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/CAUSE-ANALYSIS.md)、[活动源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[源码导读](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。所有两进程/三KV原始数据可追溯；NCU动态spill/L2/HBM/achieved occupancy UNKNOWN。

下一步先FFN1主循环/tile/资源对齐，再真实DAG补位。108是worker池，不要求每阶段凑108任务。当前M39累计1852.227/36000秒；本轮owned全退。

RoboTwin后台链已自然完成（13:02核验）：CURRENT18为13成功5自然失败，FIT5779/CAL3502行；ON固定500更新、11195行GPU adapter/decode校准齐。不是DEV成功率；teacher B2/B14 parity未测、四权限false。[原件与清理](../.pipeline/experiments/robotwin-pilot-20261020/current18-followup-20261021/RESULT.md)。DEV90入口待审查，未启动/排队；不宣称GPU仍在运行。原24h累计7.1672h。

旧paused72、FAILED/retention、baseline/原tile/LIBERO生产与四项false不变。公开包不含PT/KV/相机/权重/DSO或完整profiler/SASS；不能只凭GitHub宣称重算全部真实tensor。[前检查点](history/ffn-resource-control-20261021/CURRENT-EXECUTION-20261021.md)。
